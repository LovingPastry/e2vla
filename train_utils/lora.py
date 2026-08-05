import math
import torch
from typing import List, Sequence, Tuple
from torch import nn, Tensor
from models.layers.mha import MySimpleMHA
from models.conv_tower import CONV_BRANCH_KEYS


class LoraLinear(nn.Module):
    """`lin(x) + (x @ A) @ B`, with the base Linear frozen.

    The factors are held in float32 even when the base Linear is not. That matters for
    the vision backbones, which are loaded in bfloat16: a bf16 parameter updated by AdamW
    at lr 1e-4 loses most of the update to rounding (bf16 carries ~3 significant decimal
    digits), and a fp32 `A` against a bf16 activation is a hard dtype error in matmul.
    Both are fixed by keeping the master copy in fp32 and casting *the factors* down to
    the activation dtype in forward -- casting the activation up instead would be
    correct too, but would keep an fp32 copy of every backbone activation alive for the
    backward pass.

    Under autocast (the action-expert path) this is a no-op: autocast decides the matmul
    dtype either way, so the cast changes nothing there.
    """

    def __init__(self, lin: nn.Linear, r: int):
        super().__init__()
        self.r = r
        self.lin = lin

        for p in self.lin.parameters():
            p.requires_grad_(False)

        self.A = nn.Parameter(torch.zeros(lin.in_features, r, dtype=torch.float32,
                                          device=lin.weight.device))
        self.B = nn.Parameter(torch.zeros(r, lin.out_features, dtype=torch.float32,
                                          device=lin.weight.device))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A, math.sqrt(5))
        with torch.no_grad():
            scaling = 1.0 / self.r
            self.A.mul_(scaling)

    def forward(self, x: Tensor):
        out = self.lin(x)
        A, B = self.A.to(x.dtype), self.B.to(x.dtype)
        return out + (x @ A) @ B


def replace_with_lora_linear(model: nn.Module, r: int):
    num_replaced = 0
    for name, m in model.named_modules():
        if isinstance(m, MySimpleMHA):
            for key in ["to_q", "to_k", "to_v", "to_qk", "to_kv", "to_qkv"]:
                if hasattr(m, key):
                    num_replaced += 1
                    print("[INFO] [{}] Replace {}".format(num_replaced, name))
                    n = len(key.replace("to_", ""))
                    lora_linear = LoraLinear(getattr(m, key), r * n)
                    setattr(m, key, lora_linear)
    return model, num_replaced


# Kept trainable alongside the LoRA factors. LayerNorm affine params and biases are the
# standard companions to LoRA ("LoRA + bias/norm tuning"): they cost almost nothing and
# absorb the distribution shift that low-rank updates alone adapt to slowly. The two
# learned embeddings are tiny (64x768 and 768) and are the model's only handle on
# "which query slot" and "which camera is the main one".
#
# CONV_BRANCH_KEYS is here for a different reason: LoRA decomposes an *update* around a
# pretrained base, and the conv branch has no pretrained base inside this model -- it is
# new in this run. Freezing it would leave it at its initialisation while every log line
# still reported it as part of the model, which is precisely the silent failure this
# repo's stamps exist to prevent. So a `lora_rank > 0 + conv_tower` run adapts the trunk
# by low rank and trains the branch densely, which is the intended combination.
LORA_KEEP_TRAINABLE = (("norm", ".bias", "qformer.queries", "main_cam_embed")
                       + CONV_BRANCH_KEYS)


def setup_lora(module: nn.Module, rank: int, keep_trainable=LORA_KEEP_TRAINABLE):
    """Inject LoRA into `module`'s attention projections and freeze everything else.

    MUST be called AFTER the pretrained weights are loaded: `replace_with_lora_linear`
    wraps each `nn.Linear` in a `LoraLinear`, which renames `...to_q.weight` to
    `...to_q.lin.weight` in the state_dict. Loading a plain (non-LoRA) checkpoint into
    an already-injected model would therefore fail on every attention projection.

    Injection is a no-op at the function level: `LoraLinear.B` is zero-initialised, so
    `lin(x) + (x @ A) @ B` is bit-identical to `lin(x)` until training moves B.

    End to end the context output can still drift by ~1e-6 in float32. That is not a
    logic error: `lin(x) + 0` allocates a fresh tensor, which lands at a different
    address and can take a different blocking path through the matmul/SDPA kernels.
    Verified: the drift shrinks when the thread count is pinned to 1, parameters stay
    bit-identical, and every LoraLinear output matches its base Linear exactly. For
    scale, training runs in bfloat16, whose epsilon is ~4000x larger than this drift.

    Args:
        module: subtree to adapt (here: `actor.context_encoder`)
        rank: LoRA rank r. The effective rank of a fused projection is `r * n`, where n
            is the number of matrices it fuses (to_qkv -> 3r), so each individual
            projection gets rank r.
        keep_trainable: name substrings that stay trainable despite the freeze

    Returns:
        (num_replaced, num_trainable_params)
    """
    if rank <= 0:
        raise ValueError("lora rank must be positive, got {}".format(rank))

    module, num_replaced = replace_with_lora_linear(module, rank)

    # replace_with_lora_linear only freezes the Linears it wrapped; freeze the rest here
    # (FFNs, projections, the QFormer's non-attention weights, ...).
    lora_params = {id(p) for p in get_lora_parameters(module)}
    num_trainable = 0
    for name, p in module.named_parameters():
        if id(p) in lora_params or any(pat in name for pat in keep_trainable):
            p.requires_grad_(True)
            num_trainable += p.numel()
        else:
            p.requires_grad_(False)

    total = sum(p.numel() for p in module.parameters())
    print("[INFO] LoRA rank={}: wrapped {} attention projections; {:.2f}M / {:.2f}M "
          "params trainable in this subtree ({:.1f}%)"
          .format(rank, num_replaced, num_trainable / 1e6, total / 1e6,
                  num_trainable / total * 100))
    return num_replaced, num_trainable


def get_lora_parameters(model: nn.Module):
    params: List[nn.Parameter] = []
    for m in model.modules():
        if isinstance(m, LoraLinear):
            params.append(m.A)
            params.append(m.B)
    return params


def lora_state_dict(module: nn.Module):
    """The trainable tensors of `module` -- for a LoRA-adapted VLM, exactly the factors.

    Kept separate from `state_dict()` on purpose: the backbones are 200M+ of weights
    that are pulled from HuggingFace/torch.hub at construction time and never change, so
    serializing them into every checkpoint would multiply its size for no information.
    """
    return {name: p.detach() for name, p in module.named_parameters() if p.requires_grad}


# ---------------------------------------------------------------------------
# LoRA for the frozen VLM backbones (off by default; see TrainConfig.vlm_lora_rank)
# ---------------------------------------------------------------------------

# target name -> (submodule path inside `VLM`, names of the nn.Linear children to wrap).
#
# Paths, not a recursive search for "anything that looks like attention": the two
# backbones are third-party code (torch.hub DINOv2, HF transformers SigLIP) whose
# internals can move under us, and a silent zero-match is exactly the failure this
# should not have. `setup_vlm_lora` raises when a target matches nothing.
#
# Only the attention projections are wrapped, and the MLPs are left alone -- the same
# choice `replace_with_lora_linear` makes for the action expert.
VLM_LORA_TARGETS = {
    # DINOv2 fuses q/k/v into a single Linear, so it gets rank 3r: `replace_with_lora_linear`
    # applies the same rule to `to_qkv`, and it keeps the per-projection rank at r.
    "dinov2": ("dinov2.frozen.dinov2.dino.blocks", ("qkv",)),
    "siglip_vision": ("siglip.frozen.siglip.siglip.vision_model", ("q_proj", "k_proj", "v_proj")),
    "siglip_text": ("siglip.frozen.siglip.siglip.text_model", ("q_proj", "k_proj", "v_proj")),
}

# What `vlm_lora_targets` means when left unset: the two image towers. The text tower is
# excluded because a single-task fine-tuning set carries one prompt, so adapting the text
# encoder fits that one string rather than anything transferable.
DEFAULT_VLM_LORA_TARGETS = ("dinov2", "siglip_vision")


def _fusion_factor(linear_name: str):
    """How many projections a Linear named `linear_name` fuses ("qkv" -> 3, "q_proj" -> 1).

    Mirrors the `to_qkv -> 3r` rule in `replace_with_lora_linear`, so that "rank r" means
    the same thing -- rank r per projection -- whether or not the backbone fuses them.
    """
    head = linear_name.split("_")[0]
    if head and set(head) <= set("qkv"):
        return len(head)
    return 1


def replace_named_linears_with_lora(module: nn.Module, r: int, linear_names: Sequence[str]):
    """Wrap every `nn.Linear` child whose attribute name is in `linear_names`.

    The name-based variant of `replace_with_lora_linear`, for backbones this repo does
    not define and therefore cannot match by class.

    Args:
        module: subtree to walk
        r: LoRA rank per projection, scaled by `_fusion_factor` for fused Linears
        linear_names: attribute names to match, e.g. ("qkv",) or ("q_proj", "v_proj")

    Returns:
        int, how many Linears were wrapped
    """
    names = set(linear_names)
    # Collect first, mutate after: `named_modules()` walks the tree lazily, and swapping
    # a child in mid-walk would hand the traversal the freshly built LoraLinear.
    targets: List[Tuple[nn.Module, str, nn.Linear]] = []
    for parent in module.modules():
        if isinstance(parent, LoraLinear):
            continue  # already wrapped; its own child is called "lin"
        for child_name, child in parent.named_children():
            if child_name in names and isinstance(child, nn.Linear):
                targets.append((parent, child_name, child))

    for parent, child_name, child in targets:
        setattr(parent, child_name, LoraLinear(child, r * _fusion_factor(child_name)))
    return len(targets)


def setup_vlm_lora(vlm: nn.Module, rank: int, targets: Sequence[str] = None):
    """Inject LoRA into the frozen VLM backbones and leave everything else frozen.

    Off by default and only reachable from the config (`TrainConfig.vlm_lora_rank`).
    What it buys: the pretrained features come from web images, and a real-robot rig --
    its lighting, its gripper occluding the wrist view, its texture-less table -- is
    out of that distribution in ways the 60M-parameter ContextEncoder can only work
    around, not fix. What it costs, and this is the part worth knowing before turning it
    on: activations for both ViTs now have to be kept for the backward pass, for every
    camera of every sample in the batch, so step time and memory both go up sharply.
    Halve `bs` before assuming it will fit.

    Unlike `setup_lora`, nothing but the factors is trained: no bias/norm tuning. Norm
    statistics are the part of a pretrained backbone that is most worth keeping intact,
    and there is no pretrained-on-this-robot state to re-fit them to.

    Order, same as `setup_lora`: this renames `...q_proj.weight` to `...q_proj.lin.weight`,
    so it must run before any state_dict that was written by an adapted model is loaded.

    Args:
        vlm: the `VLM` module (`model.vlm`)
        rank: LoRA rank r per projection. Fused projections get r * (number fused).
        targets: which towers to adapt, from `VLM_LORA_TARGETS`. None -> the two image
            towers.

    Returns:
        (num_replaced, num_trainable_params)
    """
    if rank <= 0:
        raise ValueError("vlm lora rank must be positive, got {}".format(rank))

    targets = list(DEFAULT_VLM_LORA_TARGETS if targets is None else targets)
    unknown = [t for t in targets if t not in VLM_LORA_TARGETS]
    if unknown:
        raise ValueError(
            "unknown vlm_lora_targets {}; valid targets are {}"
            .format(unknown, sorted(VLM_LORA_TARGETS)))

    num_replaced = 0
    for target in targets:
        path, linear_names = VLM_LORA_TARGETS[target]
        try:
            subtree = vlm.get_submodule(path)
        except AttributeError as e:
            raise RuntimeError(
                "vlm_lora target '{}' points at '{}', which does not exist in this VLM. "
                "The backbone's module tree changed (torch.hub / transformers version); "
                "fix the path in VLM_LORA_TARGETS rather than dropping the target."
                .format(target, path)) from e

        n = replace_named_linears_with_lora(subtree, rank, linear_names)
        if n == 0:
            raise RuntimeError(
                "vlm_lora target '{}' matched no Linear named {} under '{}'. Injecting "
                "nothing would train nothing and report success, so this is an error."
                .format(target, tuple(linear_names), path))
        print("[INFO] VLM LoRA: wrapped {} Linear(s) named {} under {}"
              .format(n, tuple(linear_names), path))
        num_replaced += n

    # Freeze everything else. The backbones were already frozen at construction, so this
    # is really about the factors -- but stating it here keeps the invariant in one place
    # instead of depending on VLM.__init__ having run first.
    lora_params = {id(p) for p in get_lora_parameters(vlm)}
    num_trainable = 0
    for p in vlm.parameters():
        if id(p) in lora_params:
            p.requires_grad_(True)
            num_trainable += p.numel()
        else:
            p.requires_grad_(False)

    total = sum(p.numel() for p in vlm.parameters())
    print("[INFO] VLM LoRA rank={} targets={}: {} projections wrapped; {:.2f}M / {:.2f}M "
          "params trainable in the VLM ({:.2f}%)"
          .format(rank, targets, num_replaced, num_trainable / 1e6, total / 1e6,
                  num_trainable / total * 100))
    return num_replaced, num_trainable


@torch.no_grad()
def merge_lora_linear(model: nn.Module, inplace: bool = True):

    index = 0

    def _merge_lora_linear_inplace(model: nn.Module, target_module_name=""):
        nonlocal index
        for name, m in model.named_children():
            if isinstance(m, LoraLinear):
                index += 1
                linear = m.lin
                lora_weight = torch.einsum("i r, r o -> o i", m.A, m.B)

                if inplace:
                    linear.weight.copy_(linear.weight + lora_weight)
                    setattr(model, name, linear)
                else:
                    new_linear = nn.Linear(
                        in_features=linear.in_features,
                        out_features=linear.out_features,
                        bias=linear.bias is not None,
                        device=linear.weight.device,
                        dtype=linear.weight.dtype,
                    )
                    new_linear.weight.copy_(linear.weight + lora_weight)
                    if new_linear.bias is not None:
                        new_linear.bias.copy_(linear.bias)
                    setattr(model, name, new_linear)

                print("[INFO] [{}] Merge {}".format(index, target_module_name + "." + name))
            else:
                _merge_lora_linear_inplace(m, target_module_name + "." + name if target_module_name else name)
        return model
    
    merged = _merge_lora_linear_inplace(model, "")
    for m in merged.modules():
        assert not isinstance(m, LoraLinear)
    
    return merged


