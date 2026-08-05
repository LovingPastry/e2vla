"""A trainable, low-level convolutional branch that runs alongside the frozen ViTs.

Everything else on the vision side of this repo is a frozen, web-pretrained backbone
(`models/encoders/`). This module is the opposite: a small CNN that is fully trained on
the target robot's own data. The motivation is that a real rig -- its lighting, its
gripper occluding the wrist view, its texture-less table -- sits outside DINOv2's and
SigLIP's training distribution in ways the 60M-parameter `ContextEncoder` can only route
around, not repair. `vlm_lora_rank` is the other answer to the same problem, and it is far
more expensive: it forces both ViTs to keep their activations for the backward pass.

Only the first two stages of ResNet18 are kept (`layer3`/`layer4` are never constructed).
Those stages carry edges, texture and local colour statistics -- exactly the information a
patch-16 ViT throws away and exactly the information that is cheap to re-learn from a few
hundred demonstrations. The deep, semantic end of the network is what the frozen ViTs
already do well, so re-learning it from ~2M parameters would be strictly worse.

The tower is deliberately NOT placed under `VLM`: `train.py` only ever serializes
`model.actor.state_dict()`, so a trainable module hanging off `VLM` would never be saved,
and `setup_vlm_lora` would silently freeze it. It lives inside `ContextEncoder`, which
also gets it EMA, weight-decay grouping and the per-tensor checkpoint compat report for
free.
"""

import torch
from typing import Optional
from torch import nn, Tensor
from einops import rearrange


# What `TrainConfig.conv_tower` accepts. "none" reproduces the original architecture
# exactly -- no module is constructed and no state_dict key appears.
CONV_TOWERS = ("none", "resnet18")

# ResNet18's ImageNet statistics. `rgbs` arrives as float in [0,1] (the dataset divides by
# 255, see `data_utils/dataset_base.py`), so this is the only preprocessing needed. Same
# constants DINOv2 uses (`models/encoders/dino.py`); SigLIP's differ (0.5/0.5).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Total downsampling factor of the tower: conv1 /2, maxpool /2, layer2 /2, head /2.
# A 256x256 input therefore yields the 16x16 = 256 tokens the two ViTs also produce.
TOTAL_STRIDE = 16

# The three `ContextEncoder` attributes that exist only when a conv tower is configured,
# as name fragments. Three places need to agree on this list, so it lives here:
#   * `train.py` -- the keys a pretrained (tower-less) checkpoint is allowed to be missing
#   * `train_utils/lora.py` -- kept trainable under LoRA, since a freshly initialised
#     branch has no pretrained base to decompose a low-rank update around
#   * `models/vla.py` -- see CONV_LR_KEYS below
CONV_BRANCH_KEYS = ("conv_tower", "proj_v.2", "proj_fuse")

# The subset of the above that gets `conv_tower_lr_scale` applied. `proj_fuse` is
# deliberately excluded: it is initialised to [I | 0] so that the trunk passes through
# untouched, and running it at the branch's larger learning rate would tear up that exact
# warm start in the first few steps. It belongs to the trunk's schedule, not the branch's.
CONV_LR_KEYS = ("conv_tower", "proj_v.2")


class ResNet18Tower(nn.Module):
    """ResNet18 truncated after `layer2`, plus a stride-2 head that lands on a 16x16 grid.

    Input is the raw RGB at whatever resolution the dataset produced -- there is no
    internal resize, unlike `models/encoders/*` which each resize to their backbone's
    fixed input size. The consequence is that the output grid is `H // 16`, so the caller
    must check it against the ViT token count; `ContextEncoder` does exactly that and
    raises with an actionable message.

    Attribute names mirror torchvision's so that a checkpoint's tensor names read the same
    way as the reference implementation (`conv_tower.layer1.0.conv1.weight`).

    Args:
        out_dim: channels the head emits, i.e. the width `ContextEncoder.proj_v[2]` takes
        pretrained: load ImageNet weights. Off only for tests -- from-scratch conv weights
            on a few hundred demonstrations converge much more slowly.
    """

    def __init__(self, out_dim: int = 256, pretrained: bool = True):
        super().__init__()
        # Imported here, not at module scope: torchvision.models pulls in a large module
        # tree that nothing else in this repo needs, and `models/vla.py` imports this file
        # unconditionally even when conv_tower is "none".
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            net = resnet18(weights=weights)
        except Exception as e:
            raise RuntimeError(
                "could not build ResNet18 with ImageNet weights. torchvision downloads "
                "them from download.pytorch.org, which HF_ENDPOINT (set in envars.py) "
                "does NOT redirect. Either make that host reachable, pre-place the file "
                "under $TORCH_HOME/hub/checkpoints/, or pass pretrained=False and accept "
                "a from-scratch branch."
            ) from e

        # Stages 1-2 only. layer3/layer4/avgpool/fc are dropped by never referencing them,
        # so they are not parameters of this module and never reach the checkpoint.
        self.conv1 = net.conv1        # /2, 3 -> 64
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool    # /4
        self.layer1 = net.layer1      # /4,  64 -> 64
        self.layer2 = net.layer2      # /8, 64 -> 128

        # /16 and up to `out_dim`. bias=False because a BatchNorm follows -- and also
        # because `VLA.reset_parameters` zeroes every trainable Conv2d/Linear bias in the
        # model, which would be a silent no-op here but is confusing to reason about.
        self.head = nn.Sequential(
            nn.Conv2d(128, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )
        self.out_dim = out_dim

        # Buffers, not parameters: they must ride along in the state_dict but never be
        # trained or picked up by `parameter_groups()`.
        self.register_buffer(
            "pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer(
            "pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, rgb: Tensor):
        """
        Args:
            rgb (Tensor): (B, Ncam, 3, H, W), float in [0, 1]

        Returns:
            tokens (Tensor): (B, Ncam, (H//16)*(W//16), out_dim)
        """
        b, ncam = rgb.shape[0], rgb.shape[1]
        x = rearrange(rgb, "b n c h w -> (b n) c h w")
        x = (x - self.pixel_mean) / self.pixel_std

        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.head(x)

        # Row-major over (h, w) -- the SAME order `FrozenEncoder.encode_aux` uses for
        # `norm_xy_ds` (`models/encoders/siglip.py`, "b c h w -> b (h w) c") and the same
        # order the ViTs emit patches in. `ContextEncoder` adds `proj_pe(norm_xy_ds)` to
        # these tokens, so a transposed flatten here would attach every patch to the wrong
        # image coordinate without raising anything.
        x = rearrange(x, "(b n) c h w -> b n (h w) c", b=b, n=ncam)
        return x


def build_conv_tower(name: Optional[str], out_dim: int = 256, pretrained: bool = True):
    """`None` for "none", otherwise the tower. Mirrors `build_action_space`'s idiom."""
    if name is None or name == "none":
        return None
    if name not in CONV_TOWERS:
        raise ValueError("unknown conv_tower '{}'; valid choices are {}"
                         .format(name, list(CONV_TOWERS)))
    if name == "resnet18":
        return ResNet18Tower(out_dim=out_dim, pretrained=pretrained)
    raise AssertionError("unreachable")


def _self_test():
    """Shapes, the [0,1] contract, gradient flow, and -- the one that matters -- that the
    token order agrees with the `norm_xy_ds` the ContextEncoder adds to these tokens."""
    torch.manual_seed(0)
    tower = ResNet18Tower(out_dim=256, pretrained=False)

    b, ncam, hw = 2, 2, 256
    rgb = torch.rand(b, ncam, 3, hw, hw)
    tokens = tower(rgb)
    assert tokens.shape == (b, ncam, (hw // TOTAL_STRIDE) ** 2, 256), tokens.shape
    print("[OK] shape {} -> {}".format(tuple(rgb.shape), tuple(tokens.shape)))

    # normalization actually applied
    ones = torch.ones(1, 1, 3, 32, 32)
    x = (rearrange(ones, "b n c h w -> (b n) c h w") - tower.pixel_mean) / tower.pixel_std
    assert torch.allclose(x[0, 0], torch.full((32, 32), (1 - 0.485) / 0.229), atol=1e-6)
    print("[OK] ImageNet normalization applied to a [0,1] input")

    # gradients reach the first conv
    tokens.sum().backward()
    assert tower.conv1.weight.grad is not None
    assert tower.conv1.weight.grad.abs().sum() > 0
    print("[OK] gradient reaches conv1 ({:.4f})".format(tower.conv1.weight.grad.abs().sum()))

    # Token order -- see the comment in forward(). Two independent checks.
    #
    # (a) Direct: perturb one 16x16 block of the input and confirm the token that responds
    #     most is at the row-major index of that block. A transposed flatten passes every
    #     shape assertion above and fails only here.
    #     Differential, not absolute: a constant image is already far from zero after
    #     ImageNet normalization, and zero-padding makes the border tokens the loudest
    #     ones -- the argmax of a single forward is a corner, regardless of the probe.
    grid = hw // TOTAL_STRIDE
    row, col = 3, 11
    base = torch.zeros(1, 1, 3, hw, hw)
    probe = base.clone()
    probe[..., row * TOTAL_STRIDE:(row + 1) * TOTAL_STRIDE,
          col * TOTAL_STRIDE:(col + 1) * TOTAL_STRIDE] = 1.0
    # eval(): BN in train mode normalises over the batch, which would mix the two forwards
    order_tower = ResNet18Tower(out_dim=256, pretrained=False).eval()
    with torch.no_grad():
        delta = (order_tower(probe) - order_tower(base))[0, 0]  # (L, C)
    hot = delta.norm(dim=-1).argmax().item()
    hot_row, hot_col = divmod(hot, grid)
    # Within one cell, not exact. Token p of this stack is centred on input pixel 16p,
    # while ViT patch p covers [16p, 16p+16) and is centred on 16p+7.5 -- a half-cell
    # offset that no padding choice in a ResNet stem removes. It is deliberately left
    # alone: `proj_pe` and the attention stack absorb a half-patch shift, and the branch
    # is trained anyway. What this assertion is really for is the transposed flatten,
    # which would land at (col, row) -- eleven cells away, not one.
    assert abs(hot_row - row) <= 1 and abs(hot_col - col) <= 1, \
        "token {} = ({}, {}) lit up, expected about ({}, {})".format(
            hot, hot_row, hot_col, row, col)
    print("[OK] row-major token order (block ({},{}) -> token {} = ({},{}))"
          .format(row, col, hot, hot_row, hot_col))

    # (b) Cross-check against the module that actually produces `norm_xy_ds`. Needs the
    #     SigLIP weights on disk, so it is skipped rather than failed when they are not.
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, hw), torch.linspace(-1, 1, hw),
                            indexing="ij")
    ramp = torch.stack([xs, ys])[None, None, None]           # (1,1,1,2,H,W)
    pooled = nn.AvgPool2d(TOTAL_STRIDE)(rearrange(ramp[:, -1], "b n c h w -> (b n) c h w"))
    mine = rearrange(pooled, "(b n) c h w -> b n (h w) c", b=1, n=1)
    try:
        from .encoders.siglip import Encoder as Siglip
        ref = Siglip().pool_mv_aux(ramp)[:, -1]              # (1,1,256,2)
    except Exception as e:
        print("[SKIP] norm_xy_ds cross-check (could not build SigLIP: {})".format(e))
    else:
        assert torch.allclose(ref, mine, atol=1e-5), (ref[0, 0, :3], mine[0, 0, :3])
        print("[OK] flatten matches norm_xy_ds (max |delta| = {:.2e})"
              .format((ref - mine).abs().max()))

    n = sum(p.numel() for p in tower.parameters())
    print("[INFO] ResNet18Tower(out_dim=256): {:.3f}M parameters".format(n / 1e6))


if __name__ == "__main__":
    import envars  # noqa: F401  -- must precede torchvision, see envars.py
    _self_test()
