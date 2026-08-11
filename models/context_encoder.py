"""Frozen backbone features -> a short, fixed-length context for the diffusion head.

Four encoders live here, selected by `TrainConfig.context_encoder`. They all satisfy the
same contract -- `forward(vl_obs, vl_feature, fp16) -> (cond, cond_mask)` with
`cond` of shape (B, Lc, hdim) -- so `DiffusionHead` never learns which one it is talking
to, and swapping them is an ablation rather than a rewrite.

    name           language   camera params   params @ hdim=768   what it is
    ------------   --------   -------------   -----------------   ----------------------
    "vl"           yes        yes (PRoPE)     60.95M              the original VLA path
    "sa"           NO         NO              57.40M              same topology, language
                                                                  cross-attn -> a second
                                                                  self-attention
    "transformer"  NO         NO              12.42M              one self-attention
                                                                  stack, then pooling
    "mlp"          NO         NO               4.14M              no attention at all

The three vision-action (VA) variants exist to answer two separate questions that the
original architecture bundles together:

  * "sa" asks whether the *language* stream was doing any work, holding the capacity
    fixed. Everything the language tokens used to be attended to is now attended to
    inside the vision stream itself, so a drop in success rate is attributable to the
    missing instruction rather than to a smaller network.
  * "transformer" / "mlp" ask how much of the 61M-parameter fusion stack a single-task
    policy actually needs. Both keep the projection stem and the 64-token output; what
    they remove is the multi-stage attention between them.

WHAT "NO CAMERA PARAMS" MEANS, because it is load-bearing and quiet if you get it wrong.
The original encoder consumes calibration twice: `proj_pe(norm_xy_ds)` needs the
intrinsics K, and PRoPE needs the extrinsics. The VA variants replace the first with a
fixed [-1, 1] grid over the patch layout (an affine reparameterisation of the same
quantity for an ideal pinhole, and `proj_pe` is a learned MLP that absorbs the affine)
and drop the second outright. The knock-on effect is on the *action space*, not on this
file: with no extrinsics there is no camera frame to express actions in, so a VA run must
use `action_space="ee_base"` (world/base-frame deltas) or a joint space.
`ActionExpert.__init__` refuses the inconsistent combination.
"""

import torch
import torch.nn.functional as F

from einops import rearrange
from torch import nn, Tensor
from typing import Dict, Optional

from .dit import DiT
from .qformer import QFormerITM, QFormerVision
from .conv_tower import build_conv_tower, TOTAL_STRIDE as CONV_TOTAL_STRIDE
from .layers.utils import simple_mlp, concat_mask
from .layers.pe import se3_inverse
from .layers.attn_dn import FFWSelfAttentionLayers, init_xncoder


# What `TrainConfig.context_encoder` accepts. Single source of truth: `configs.py`
# validates against it and `train_utils/ckpt.py` stamps the chosen value into every
# checkpoint.
CONTEXT_ENCODERS = ("vl", "sa", "transformer", "mlp")

# What a checkpoint carrying no "context_encoder" key is. Everything written before this
# option existed is the vision-language encoder.
DEFAULT_CONTEXT_ENCODER = "vl"

# Length of the context the head is conditioned on. Fixed across all four variants: it is
# the one number `DiffusionHead`'s cost scales with, so keeping it constant is what makes
# the parameter counts above comparable.
NUM_CONTEXT_QUERIES = 64


def pool_tokens(x: Tensor, num_out: int):
    """(B, L, C) -> (B, num_out, C) by average pooling over the token axis.

    The parameter-free stand-in for the QFormer, used by the two small variants. Tokens
    arrive camera-major and row-major within a camera (`b (n l) c`, and `l` itself is
    `(h w)`), so a group is a run of consecutive patches along an image row -- with the
    16x16 grids both ViTs produce and the usual two cameras, exactly half a row each.
    A group can straddle two cameras when Ncam does not divide `num_out`; that costs one
    blurred token out of 64 and is not worth a per-camera reshape that would make the
    output length depend on the camera count.
    """
    if x.shape[1] == num_out:
        return x
    return F.adaptive_avg_pool1d(x.transpose(1, 2), num_out).transpose(1, 2)


class ContextEncoderBase(nn.Module):
    """The projection stem every variant shares, plus the contract they all implement.

    Held in a base class rather than a submodule on purpose: `self.proj_v` here is
    `context_encoder.proj_v.*` in the state_dict either way, and a submodule would have
    renamed every tensor of every released checkpoint.

    Subclasses build their own fusion stack in `__init__`, call `reset_stem_parameters()`
    from their own `reset_parameters()`, and implement `forward`.

    Two class attributes drive everything outside this file:
      * `uses_language`      -- `VLA` skips the SigLIP text tower entirely when False
      * `uses_camera_params` -- `VLA` passes None for intrinsics/extrinsics when False,
                                and `ActionExpert` then refuses a camera-relative action
                                space
    """

    uses_language: bool = False
    uses_camera_params: bool = False

    def __init__(self, hdim: int, conv_tower: Optional[str] = None, conv_dim: int = 256):
        super().__init__()
        self.hdim = hdim
        # THE ORDER OF THESE THREE CALLS IS LOAD-BEARING, and not for the state_dict --
        # that is keyed by name. `ExponentialMovingAverage` holds one flat list matched to
        # `model.parameters()` **positionally**, and `optimizer.load_state_dict` matches
        # the tensors inside a param group the same way. Registering a module earlier or
        # later than the original `ContextEncoder` did would therefore make every
        # pre-existing EMA shadow and every `-c` resume land on the wrong tensors --
        # silently, since the shapes still line up. Hence the hook: "vl" registers
        # `proj_l` between the vision projections and the 2D PE, exactly where it was.
        self._build_vision_stem(conv_tower, conv_dim)
        self._build_extra_stem()
        self._build_pe_stem()

    def _build_vision_stem(self, conv_tower: Optional[str], conv_dim: int):
        hdim = self.hdim
        self.proj_v = nn.ModuleList([
            simple_mlp([768, hdim, hdim], ln=True),  # for dinov2 vision embeds
            simple_mlp([768, hdim, hdim], ln=True),  # for siglip vision embeds
        ])
        # Appended as index 2 so `proj_v.0.*` / `proj_v.1.*` keep their names: a pretrained
        # checkpoint still matches them tensor for tensor.
        self.conv_tower = build_conv_tower(conv_tower, out_dim=conv_dim)
        if self.conv_tower is not None:
            self.proj_v.append(simple_mlp([conv_dim, hdim, hdim], ln=True))
            self.proj_fuse = nn.Linear(2 * hdim, hdim)
        else:
            self.proj_fuse = None

    def _build_extra_stem(self):
        """Hook for modules that must be registered here rather than in the subclass's
        own `__init__`, i.e. between the vision projections and the 2D PE. Only "vl" uses
        it, and only to keep `proj_l` at its historical position -- see `__init__`."""

    def _build_pe_stem(self):
        self.proj_pe = simple_mlp([2, self.hdim, self.hdim], ln=True)  # normalized coords
        self.main_cam_embed = nn.Parameter(torch.zeros(self.hdim))

    def reset_stem_parameters(self):
        # zero both weight and bias, otherwise pe2d is a random constant offset at init
        nn.init.zeros_(self.proj_pe[-1].weight)
        nn.init.zeros_(self.proj_pe[-1].bias)

        if self.proj_fuse is not None:
            # [I | 0]: at init the fusion returns the ViT sum untouched, so a model with a
            # conv tower is functionally identical to one without and a pretrained trunk
            # warm-starts exactly. Same trick as proj_pe above and as LoRA's zero-init B.
            # It is strictly more expressive than the sum it starts as -- summing the two
            # streams is the special case [I | I].
            hdim = self.proj_fuse.out_features
            with torch.no_grad():
                self.proj_fuse.weight.copy_(torch.cat(
                    [torch.eye(hdim), torch.zeros(hdim, hdim)], dim=1))
                self.proj_fuse.bias.zero_()

    def project_vision(self, vl_obs: Dict[str, Tensor], vl_feature: Dict[str, Tensor],
                       fp16: bool):
        """Both frozen towers (and optionally the conv branch) -> (B, Ncam, Lv, hdim)."""
        x_v: Tensor = self.proj_v[0](vl_feature["vision_embeds"][0]) + \
                      self.proj_v[1](vl_feature["vision_embeds"][1])  # (B, Ncam, Lv, C)
        if self.conv_tower is None:
            return x_v

        # Checked before the tower runs, not after: the token count is fully decided
        # by the image size and the tower's fixed stride, and the conv stack is the
        # most expensive thing in this forward -- no reason to pay for it first.
        # A mismatch here is not a broadcast bug waiting to happen (`cat` would raise
        # too), it is a misconfigured `output_image_hw`, so say so.
        img_h, img_w = vl_obs["rgb"].shape[-2:]
        num_conv_tok = (img_h // CONV_TOTAL_STRIDE) * (img_w // CONV_TOTAL_STRIDE)
        if num_conv_tok != x_v.shape[-2]:
            raise RuntimeError(
                "conv_tower would produce {} tokens per camera but the ViTs produced "
                "{}. The tower downsamples by {}x and does not resize its input, so "
                "the dataset's output_image_hw must be {}x the ViT patch grid "
                "(16x16), i.e. (256, 256). Got rgb {}x{}."
                .format(num_conv_tok, x_v.shape[-2], CONV_TOTAL_STRIDE,
                        CONV_TOTAL_STRIDE, img_h, img_w))
        # The trainable branch reads the raw pixels, not the frozen features, and only
        # the latest frame -- matching what `VLM` hands over in `vision_embeds`.
        # Autocast for the same reason pre_attn/post_attn have it: proj_v runs outside
        # those blocks, so without it the conv stack would run in fp32.
        with torch.autocast(
            x_v.device.type,
            torch.bfloat16 if fp16 else torch.float32
        ):
            x_cnn = self.proj_v[2](self.conv_tower(vl_obs["rgb"][:, -1]))
            x_v = self.proj_fuse(torch.cat([x_v, x_cnn.to(x_v.dtype)], dim=-1))
        # Back to fp32, which is what the tower-less path hands to `x_v + pe2d`.
        # Keeping the dtypes identical is what makes the [I | 0] warm start exact
        # rather than merely close.
        return x_v.float()

    def token_xy(self, x_v: Tensor, vl_feature: Dict[str, Tensor]):
        """Normalized 2D coordinate of every patch token, (B, Ncam, Lv, 2).

        With calibration that is `norm_xy_ds`, the pixel grid pushed through K^-1 --
        metrically meaningful, and the reason two cameras with different fields of view
        get different encodings. Without it, a fixed [-1, 1] grid over the patch layout.

        The substitution is safe *here* and nowhere else: `proj_pe` is a learned MLP whose
        input is a 2-vector per token, and for an ideal pinhole the two grids differ by a
        per-camera affine, which the MLP's first Linear can represent. What is genuinely
        lost is the FOV difference between cameras -- with a fixed grid, a wide-angle and
        a narrow view get identical position codes. Nothing metric is computed from this,
        so that degrades the encoding rather than corrupting it.
        """
        if self.uses_camera_params:
            return vl_feature["norm_xy_ds"]

        B, num_cam, num_patch, _ = x_v.shape
        grid = int(round(num_patch ** 0.5))
        if grid * grid != num_patch:
            raise RuntimeError(
                "the camera-parameter-free 2D positional encoding assumes a square patch "
                "grid, but the backbones produced {} tokens per camera. Both ViTs are "
                "configured for a 16x16 grid (256 tokens); if that changed, pass real "
                "intrinsics and use context_encoder='vl' instead."
                .format(num_patch))
        # Row-major to match the "b c h w -> b (h w) c" flatten both encoders do, so
        # token i sits at (y=i//grid, x=i%grid).
        lin = torch.linspace(-1.0, 1.0, grid, device=x_v.device, dtype=x_v.dtype)
        yy, xx = torch.meshgrid(lin, lin, indexing="ij")
        xy = torch.stack([xx, yy], dim=-1).reshape(1, 1, num_patch, 2)
        return xy.expand(B, num_cam, num_patch, 2)

    def add_pe2d(self, x_v: Tensor, vl_feature: Dict[str, Tensor]):
        """(B, Ncam, Lv, C) -> (B, Ncam*Lv, C) with the additive 2D position code.

        Flattening the cameras into the token axis is what lets the fusion stack attend
        over all views jointly.
        """
        pe2d = self.proj_pe(self.token_xy(x_v, vl_feature))  # (B, Ncam, Lv, C)
        x_v = rearrange(x_v, "b n l c -> b (n l) c")
        pe2d = rearrange(pe2d, "b n l c -> b (n l) c")
        return x_v + pe2d

    def tag_main_cam(self, x_v: Tensor, num_cam: int):
        """Mark camera 0's tokens, (B, Ncam*Lv, C) in and out.

        Actions live in camera 0's frame under "ee_cam", and camera 0 is the third-person
        view under every action space, so the head must be able to tell which block of
        tokens it is. clone() because the caller's tensor may be a view and the next line
        writes in place.
        """
        x_v = x_v.clone()
        num_patch_flat = x_v.shape[1] // num_cam
        x_v[:, :num_patch_flat] = x_v[:, :num_patch_flat] + self.main_cam_embed
        return x_v

    def forward(self, vl_obs: Dict[str, Tensor], vl_feature: Dict[str, Tensor],
                fp16: bool):
        """
        Args:
            vl_obs (Dict[str, Tensor]):
                - rgb: (B, To, ncam, 3, H, W)
                - norm_xy: (B, To, ncam, 2, H, W) or None, coordinates in the normalized
                  camera plane
                - text: List (length=B) of prompt, or None
                - extrinsics: (B, To, ncam, 4, 4) or None, ^{world}_{camera} T

            vl_feature (Dict[str, Tensor]):
                - norm_xy_ds: (B, Ncam, Lv, 2) or None
                - vision_embeds: List (length=num_layer) of (B, Ncam, Lv, C)
                - lang_embeds: List (length=num_layer) of (B, La, C), or None
                - lang_mask: (B, La) or None
                - extrinsics: (B, Ncam, 4, 4) or None

            fp16: if True, use bfloat16

        Returns
        -------
            context: (B, Lc, hdim)
            context_mask: (B, Lc) or None
        """
        raise NotImplementedError


class VLContextEncoder(ContextEncoderBase):
    """The original vision-language encoder. Unchanged; `context_encoder="vl"`.

    Pipeline: project both vision backbones into hdim and sum -> add the 2D coordinate
    PE -> `pre_attn` (self-attn over all cameras' patches, cross-attn to language, with
    camera-pose PRoPE) -> QFormer compresses Ncam*Lv patches to 64 queries -> `post_attn`.

    The compression is not optional: with Ncam cameras at Lv patches each, feeding the
    raw tokens to every denoising step of the head would dominate inference cost.

    Three positional encodings do three different jobs here:
      * `proj_pe(norm_xy)` -- additive, absolute position on the image plane.
        zero-initialised so it does not perturb the frozen features early in training.
      * PRoPE(extrinsics)  -- multiplicative, applied inside attention; makes attention
        between two patches depend on the *relative* pose of their cameras.
      * `main_cam_embed`   -- marks camera 0, the frame actions are expressed in.

    `conv_tower` optionally adds a third, *trainable* vision stream (see
    `models/conv_tower.py`) which is concatenated with the summed ViT stream and fused by
    `proj_fuse`. The frozen sum is left exactly as it was: the dino+siglip relationship is
    what a pretrained trunk learned, so the new modality joins from the side rather than
    inside it.
    """

    uses_language = True
    uses_camera_params = True

    def __init__(self, hdim: int, num_heads: int, num_layers: int,
                 conv_tower: Optional[str] = None, conv_dim: int = 256,
                 num_queries: int = NUM_CONTEXT_QUERIES):
        super().__init__(hdim, conv_tower=conv_tower, conv_dim=conv_dim)
        self.pre_attn = DiT(hdim, num_heads, num_layers//2, use_adaln=False,
                            pe_type="prope")  # actually, it is not a DiT but self-cross attention module
        self.qformer = QFormerITM(hdim, num_heads, num_layers=1, num_queries=num_queries)
        self.post_attn = FFWSelfAttentionLayers(hdim, num_heads, num_layers//2, use_adaln=False,
                                                bias=True, qk_norm=True, ffn_expansion=2)
        self.reset_parameters()

    def _build_extra_stem(self):
        # Registered from the base `__init__`, between the vision projections and the 2D
        # PE, because that is where the original `ContextEncoder` put it. See the note in
        # `ContextEncoderBase.__init__` for why the position matters.
        self.proj_l = nn.ModuleList([
            simple_mlp([768, self.hdim, self.hdim], ln=True)  # siglip language embeds
        ])

    def reset_parameters(self):
        init_xncoder(self.post_attn.num_layers, self.post_attn)
        self.reset_stem_parameters()

    def forward(self, vl_obs, vl_feature, fp16: bool):
        """See `ContextEncoderBase.forward`. Returns (B, Ncam*Lt, hdim) and its mask."""
        batch_size, _, num_cam, _, _, _ = vl_obs["rgb"].shape
        obs_extrinsics = vl_obs["extrinsics"]  # (B, To, Ncam, 4, 4)

        # Rebase every camera pose onto camera 0 at the latest timestep. PRoPE only ever
        # uses relative poses, so the absolute world origin must not leak in -- otherwise
        # the model overfits to each dataset's arbitrary world frame.
        cam0_extr_ref = torch.inverse(obs_extrinsics[:, -1:, 0:1]) @ obs_extrinsics  # (B, To, Ncam, 4, 4)
        x_v = self.project_vision(vl_obs, vl_feature, fp16)  # (B, Ncam, Lv, C)
        x_l: Tensor = self.proj_l[0](vl_feature["lang_embeds"][0])    # (B, Ncam, La, C)
        mask_l = vl_feature["lang_mask"]  # (B, La)

        # camera pose as multiplicative positional encoding (PRoPE)
        num_patch = x_v.shape[-2]
        extrinsic_wcT = cam0_extr_ref[:, -1]  # (B, Ncam, 4, 4), select the latest frame
        # Invert once here, on (B, Ncam, 4, 4). PRoPE needs the inverse for every query
        # token in every layer; inverting after the expand would redo the same Ncam
        # matrices Lv times per layer. se3_inverse is the analytic rigid-transform
        # inverse -- valid because cam0_extr_ref is a product of rigid transforms.
        extrinsic_cwT = se3_inverse(extrinsic_wcT)  # (B, Ncam, 4, 4)

        def expand_to_tokens(extr: Tensor):
            """(B, Ncam, 4, 4) -> (B, Ncam*Lv, 4, 4): every patch inherits its camera's
            pose. `expand` keeps this a view, so no Lv-fold memory blowup."""
            extr = extr[:, :, None, :, :].expand(batch_size, num_cam, num_patch, 4, 4)
            return rearrange(extr, "b n l r c -> b (n l) r c")

        extrinsic_pe = expand_to_tokens(extrinsic_wcT)
        extrinsic_pe_inv = expand_to_tokens(extrinsic_cwT)

        # flatten cameras into the token axis, with the additive 2D positional encoding
        x_v = self.add_pe2d(x_v, vl_feature)

        # SA before qformer
        with torch.autocast(
            x_v.device.type,
            torch.bfloat16 if fp16 else torch.float32
        ):
            x_v: Tensor = self.pre_attn(
                x=x_v,
                x_pe=extrinsic_pe,
                x_mask=None,  # every vision patch is valid
                conds=[x_l],
                cond_masks=[mask_l],
                films=None,
                x_pe_inv=extrinsic_pe_inv
            )

        # tag camera 0's tokens: actions live in its frame, so the head must be able to
        # tell which view defines "forward".
        x_v = self.tag_main_cam(x_v, num_cam)

        with torch.autocast(
            x_v.device.type,
            torch.bfloat16 if fp16 else torch.float32
        ):
            query, x_l, _ = self.qformer(
                x_vision=x_v,
                mask_vision=None,
                x_text=x_l,
                mask_text=mask_l,
            )

            query = self.post_attn(
                query=query,
            )[-1]

        cond = torch.cat([query, x_l], dim=1)
        cond_mask = concat_mask(mask0=None, mask1=mask_l,
                                L0=query.shape[1], L1=x_l.shape[1])
        return cond, cond_mask


class SelfAttnContextEncoder(ContextEncoderBase):
    """`context_encoder="sa"`: the same topology with the language stream removed.

    Every stage of the vision-language encoder survives, and each one that used to read
    the text tokens now reads the vision tokens instead:

        pre_attn   self-attn -> cross-attn to language   =>  self-attn -> self-attn
                   (`DiTBlock` with c=None; the CrossAttentionLayer keeps its own Q and
                   KV projections, so the parameter count of this stage is identical)
        qformer    64 queries cross-attending to the patches, with the query/text
                   self-attention collapsing to query self-attention (`QFormerVision`)
        post_attn  unchanged

    Two things are gone: `proj_l` (the 768 -> hdim language projection, 1.18M) and the
    QFormer's `ffn_text` (2.36M). That is the whole 3.5M difference from "vl" at
    hdim=768 -- 5.8% -- which is what "参数量基本不变" means here. Nothing else about the
    capacity or the depth changes, so a rollout difference against "vl" measures the
    instruction, not the network size.

    PRoPE is also dropped (`pe_type="rope"` with no `x_pe` passed is a no-op, and both are
    parameter-free), so this variant reads no calibration at all.
    """

    uses_language = False
    uses_camera_params = False

    def __init__(self, hdim: int, num_heads: int, num_layers: int,
                 conv_tower: Optional[str] = None, conv_dim: int = 256,
                 num_queries: int = NUM_CONTEXT_QUERIES):
        super().__init__(hdim, conv_tower=conv_tower, conv_dim=conv_dim)
        # pe_type is parameter-free either way; "rope" with x_pe=None simply applies no
        # multiplicative PE, which is what "no extrinsics" has to mean.
        self.pre_attn = DiT(hdim, num_heads, num_layers//2, use_adaln=False,
                            pe_type="rope")
        self.qformer = QFormerVision(hdim, num_heads, num_layers=1, num_queries=num_queries)
        self.post_attn = FFWSelfAttentionLayers(hdim, num_heads, num_layers//2, use_adaln=False,
                                                bias=True, qk_norm=True, ffn_expansion=2)
        self.reset_parameters()

    def reset_parameters(self):
        init_xncoder(self.post_attn.num_layers, self.post_attn)
        self.reset_stem_parameters()

    def forward(self, vl_obs, vl_feature, fp16: bool):
        """See `ContextEncoderBase.forward`. Returns (B, 64, hdim) and None."""
        _, _, num_cam, _, _, _ = vl_obs["rgb"].shape

        x_v = self.project_vision(vl_obs, vl_feature, fp16)  # (B, Ncam, Lv, C)
        x_v = self.add_pe2d(x_v, vl_feature)                 # (B, Ncam*Lv, C)

        with torch.autocast(
            x_v.device.type,
            torch.bfloat16 if fp16 else torch.float32
        ):
            x_v: Tensor = self.pre_attn(
                x=x_v,
                x_pe=None,      # no extrinsics -> no PRoPE
                x_mask=None,    # every vision patch is valid
                conds=None,     # no language -> the cross-attn re-attends to x
                cond_masks=None,
                films=None,
            )

        x_v = self.tag_main_cam(x_v, num_cam)

        with torch.autocast(
            x_v.device.type,
            torch.bfloat16 if fp16 else torch.float32
        ):
            query = self.qformer(x_vision=x_v, mask_vision=None)
            query = self.post_attn(query=query)[-1]

        # Every context token is a query slot, and all of them are valid -- no mask.
        return query, None


class TransformerContextEncoder(ContextEncoderBase):
    """`context_encoder="transformer"`: one self-attention stack, then pooling.

    projection stem -> +2D PE -> tag camera 0 -> `attn` (self-attn + FFN, `num_layers//4`
    layers over ALL cameras' patches) -> average-pool the token axis down to 64.

    The three-stage fusion of "vl"/"sa" -- pre_attn, a learned-query QFormer, post_attn --
    collapses to a single stack, and the learned compression collapses to a mean. What
    survives is the part a single-task policy plausibly needs: patches from both views
    talking to each other once, at full resolution. 12.4M parameters at hdim=768, a 4.9x
    reduction, with the same 64-token output the head already expects.

    Pooling AFTER the attention rather than before is the one design choice worth stating:
    pooling first would cost 8x less attention but would fix the spatial resolution the
    stack can reason at before it has reasoned at all.
    """

    uses_language = False
    uses_camera_params = False

    def __init__(self, hdim: int, num_heads: int, num_layers: int,
                 conv_tower: Optional[str] = None, conv_dim: int = 256,
                 num_queries: int = NUM_CONTEXT_QUERIES):
        super().__init__(hdim, conv_tower=conv_tower, conv_dim=conv_dim)
        self.num_queries = num_queries
        # num_layers is `num_actor_context_layers` (8), which "vl" splits 4/1/4 across its
        # three stages. A quarter of it keeps this variant firmly in "much smaller" range
        # while still being a transformer rather than a single attention op.
        num_attn_layers = max(1, num_layers // 4)
        self.attn = FFWSelfAttentionLayers(hdim, num_heads, num_attn_layers,
                                           use_adaln=False, bias=True, qk_norm=True,
                                           ffn_expansion=2)
        self.out_norm = nn.LayerNorm(hdim)
        self.reset_parameters()

    def reset_parameters(self):
        init_xncoder(self.attn.num_layers, self.attn)
        self.reset_stem_parameters()

    def forward(self, vl_obs, vl_feature, fp16: bool):
        """See `ContextEncoderBase.forward`. Returns (B, 64, hdim) and None."""
        _, _, num_cam, _, _, _ = vl_obs["rgb"].shape

        x_v = self.project_vision(vl_obs, vl_feature, fp16)  # (B, Ncam, Lv, C)
        x_v = self.add_pe2d(x_v, vl_feature)                 # (B, Ncam*Lv, C)
        x_v = self.tag_main_cam(x_v, num_cam)

        with torch.autocast(
            x_v.device.type,
            torch.bfloat16 if fp16 else torch.float32
        ):
            x_v = self.attn(query=x_v)[-1]
            query = self.out_norm(pool_tokens(x_v, self.num_queries))

        return query, None


class MLPContextEncoder(ContextEncoderBase):
    """`context_encoder="mlp"`: no attention anywhere.

    projection stem -> +2D PE -> tag camera 0 -> per-token MLP -> average-pool to 64.

    4.14M parameters at hdim=768, a 15x reduction, and the floor this whole family is
    measured against: every token is processed independently, so the only mixing between
    patches (and between the two cameras) is the mean inside `pool_tokens`. If this
    matches "vl" on a task, that task's context did not need spatial reasoning and the
    61M-parameter stack was buying nothing.

    The head is not similarly weakened -- `DiffusionHead` still cross-attends to these 64
    tokens over 4 DiT layers -- so this is an ablation of the *encoder*, not of the model.
    """

    uses_language = False
    uses_camera_params = False

    def __init__(self, hdim: int, num_heads: int, num_layers: int,
                 conv_tower: Optional[str] = None, conv_dim: int = 256,
                 num_queries: int = NUM_CONTEXT_QUERIES):
        super().__init__(hdim, conv_tower=conv_tower, conv_dim=conv_dim)
        # num_heads / num_layers are accepted and ignored: `build_context_encoder` hands
        # every variant the same arguments, and an MLP has neither.
        self.num_queries = num_queries
        self.token_mlp = simple_mlp([hdim, hdim, hdim], ln=True)
        self.out_norm = nn.LayerNorm(hdim)
        self.reset_parameters()

    def reset_parameters(self):
        self.reset_stem_parameters()

    def forward(self, vl_obs, vl_feature, fp16: bool):
        """See `ContextEncoderBase.forward`. Returns (B, 64, hdim) and None."""
        _, _, num_cam, _, _, _ = vl_obs["rgb"].shape

        x_v = self.project_vision(vl_obs, vl_feature, fp16)  # (B, Ncam, Lv, C)
        x_v = self.add_pe2d(x_v, vl_feature)                 # (B, Ncam*Lv, C)
        x_v = self.tag_main_cam(x_v, num_cam)

        with torch.autocast(
            x_v.device.type,
            torch.bfloat16 if fp16 else torch.float32
        ):
            x_v = self.token_mlp(x_v)
            query = self.out_norm(pool_tokens(x_v, self.num_queries))

        return query, None


CONTEXT_ENCODER_CLASSES = {
    "vl": VLContextEncoder,
    "sa": SelfAttnContextEncoder,
    "transformer": TransformerContextEncoder,
    "mlp": MLPContextEncoder,
}


def build_context_encoder(
    name: Optional[str],
    hdim: int,
    num_heads: int,
    num_layers: int,
    conv_tower: Optional[str] = None,
    conv_dim: int = 256,
    num_queries: int = NUM_CONTEXT_QUERIES,
) -> ContextEncoderBase:
    """"vl" | "sa" | "transformer" | "mlp" -> the encoder. None means "vl" (historical)."""
    if name is None or name == "":
        name = DEFAULT_CONTEXT_ENCODER
    if name not in CONTEXT_ENCODER_CLASSES:
        raise ValueError("unknown context_encoder '{}'; valid choices are {}"
                         .format(name, list(CONTEXT_ENCODERS)))
    return CONTEXT_ENCODER_CLASSES[name](
        hdim=hdim, num_heads=num_heads, num_layers=num_layers,
        conv_tower=conv_tower, conv_dim=conv_dim, num_queries=num_queries)


def count_parameters():
    """Parameter count of each variant at the `base` size. `python -m models.context_encoder`."""
    hdim, num_heads, num_layers = 768, 12, 8
    print("{:>14} {:>12} {:>10} {:>10}".format("variant", "params", "language", "camera"))
    for name in CONTEXT_ENCODERS:
        enc = build_context_encoder(name, hdim, num_heads, num_layers)
        num = sum(p.numel() for p in enc.parameters() if p.requires_grad)
        print("{:>14} {:>11.2f}M {:>10} {:>10}"
              .format(name, num / 1e6, str(enc.uses_language), str(enc.uses_camera_params)))


if __name__ == "__main__":
    count_parameters()
