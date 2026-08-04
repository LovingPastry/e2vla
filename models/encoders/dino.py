import os
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, Tensor
from torchvision.transforms import v2


class Dinov2Encoder(nn.Module):
    """Frozen DINOv2 (ViT-B/14 + 4 register tokens), loaded through ``torch.hub``.

    This is the same network as the HuggingFace ``facebook/dinov2-with-registers-base``
    checkpoint this class used to load -- that repo is a conversion of the upstream
    ``dinov2_vitb14_reg`` weights -- but it goes to the original source instead of the
    HF hub. Two consequences worth knowing:

    * ``HF_ENDPOINT`` (the mirror set in ``envars.py``) does NOT apply. torch.hub pulls
      the code from GitHub and the weights from ``dl.fbaipublicfiles.com``. Behind a
      firewall, set ``DINOV2_HUB_SOURCE=local`` and ``DINOV2_HUB_REPO=/path/to/dinov2``
      after cloning the repo yourself, and pre-place the ``.pth`` under ``$TORCH_HOME``.
    * The upstream implementation uses xformers' memory-efficient attention when it is
      installed and a plain matmul+softmax otherwise; it does not go through PyTorch
      SDPA. Without xformers this is somewhat slower than the HF path was, but the
      encoder is frozen and runs under ``no_grad``, so it only costs forward time.

    Preprocessing constants are inlined rather than read from a processor config: the
    old code overrode the processor's resize-then-center-crop with a direct resize to
    ``crop_size`` anyway, and DINOv2 uses standard ImageNet statistics.
    """

    HUB_REPO = os.environ.get("DINOV2_HUB_REPO", "facebookresearch/dinov2")
    HUB_SOURCE = os.environ.get("DINOV2_HUB_SOURCE", "github")
    HUB_MODEL = os.environ.get("DINOV2_HUB_MODEL", "dinov2_vitb14_reg")

    # ImageNet-1k statistics, what DINOv2 was trained with
    IMAGE_MEAN = (0.485, 0.456, 0.406)
    IMAGE_STD = (0.229, 0.224, 0.225)
    INPUT_SIZE = (224, 224)  # 224 / 14 -> a 16x16 patch grid

    def __init__(self):
        super().__init__()

        run_fp16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.dtype = torch.bfloat16 if run_fp16 else torch.float32

        self.dino = torch.hub.load(
            self.HUB_REPO,
            self.HUB_MODEL,
            source=self.HUB_SOURCE,
            pretrained=True,
            trust_repo=True,  # skip the interactive prompt; ignored when source="local"
        )
        self.dino = self.dino.to(self.dtype).eval()

        self.num_regs = self.dino.num_register_tokens  # 4 for the *_reg variants
        self.image_tform = v2.Compose([
            v2.Resize(size=self.INPUT_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
            v2.Normalize(mean=list(self.IMAGE_MEAN), std=list(self.IMAGE_STD)),
        ])

    @property
    def input_size(self):
        return self.INPUT_SIZE

    @property
    def output_size(self):
        return (
            self.INPUT_SIZE[0] // self.patch_size,
            self.INPUT_SIZE[1] // self.patch_size,
        )  # (16, 16)

    @property
    def patch_size(self):
        return self.dino.patch_size  # 14

    def encode_images(self, rgb: Tensor):
        """Args: rgb (B, 3, H, W) float in [0, 1]. Returns (patch tokens, CLS token)."""
        pixel_values: Tensor = self.image_tform(rgb)
        # Unlike HF, the upstream module does not cast its input for you: the patch-embed
        # conv would raise on an fp32 input against bf16 weights.
        pixel_values = pixel_values.to(self.dtype)

        # forward_features already splits off the CLS and register tokens, so no manual
        # `x[:, 1 + num_regs:]` slice is needed. `x_norm_clstoken` is the post-LayerNorm
        # CLS token, which is exactly what HF returned as `pooler_output` -- Dinov2Model
        # has no learned pooler.
        outputs = self.dino.forward_features(pixel_values)

        x_patchtokens: Tensor = outputs["x_norm_patchtokens"]  # (B, N_patch, C)
        gx: Tensor = outputs["x_norm_clstoken"]  # (B, C)
        return x_patchtokens, gx


class FrozenEncoder(nn.Module):
    def __init__(self, pool_ks: int = 1):
        super().__init__()
        self.dinov2 = Dinov2Encoder()
        self.resize2 = v2.Resize(self.dinov2.input_size, v2.InterpolationMode.BILINEAR)
        self.resize0 = v2.Resize(self.dinov2.input_size, v2.InterpolationMode.NEAREST)

        self.pool_ks = pool_ks
        self.pool_aux = nn.AvgPool2d(self.dinov2.patch_size * pool_ks)
        self.pool_token = nn.Identity() if pool_ks == 1 else nn.AvgPool2d(pool_ks)
    
    def encode_images(self, rgb: Tensor):
        x_dinov2, gx_dinov2 = self.dinov2.encode_images(rgb)
        x_dinov2 = x_dinov2.float()
        gx_dinov2 = gx_dinov2.float()

        if self.pool_ks > 1:
            h, w = self.dinov2.output_size
            x_dinov2 = rearrange(x_dinov2, "b (h w) c -> b c h w", h=h, w=w)
            x_dinov2 = self.pool_token(x_dinov2)
            x_dinov2 = rearrange(x_dinov2, "b c h w -> b (h w) c")
        return x_dinov2, gx_dinov2

    def encode_aux(self, a: Tensor):
        B, C, H, W = a.shape
        assert H == W
        
        bool_type = a.dtype == torch.bool
        float_type = a.is_floating_point()
        int_type = (not bool_type) and (not float_type)
        
        if float_type:
            a_resize = self.resize2(a)
        else:
            a_resize = self.resize0(a.float())
        
        a_ds: Tensor = self.pool_aux(a_resize)
        if bool_type:
            a_ds = a_ds > 1e-4
        elif int_type:
            a_ds = a_ds.to(a.dtype)
        a_ds = rearrange(a_ds, "b c h w -> b (h w) c")
        return a_ds


class Encoder(nn.Module):
    def __init__(self, pool_ks: int = 1):
        super().__init__()
        self.frozen = FrozenEncoder(pool_ks)
        for p in self.frozen.parameters():
            p.requires_grad_(False)
    
    @property
    def output_dim(self):
        return 768
    
    def encode_mv_images(self, rgb: Tensor):
        """
        Args:
            rgb (Tensor): (B, T, N, 3, H, W)

        Returns:
            x_ds (Tensor): (B, T, N, L, C)
            gx (Tensor): (B, T, N, C)
        """
        B, T, N, C, H, W = rgb.shape
        rgb = rearrange(rgb, "b t n c h w -> (b t n) c h w")
        with torch.no_grad():
            x_ds, gx = self.frozen.encode_images(rgb)
        x_ds = rearrange(x_ds, "(b t n) l c -> b t n l c", b=B, t=T, n=N)
        gx = rearrange(gx, "(b t n) c -> b t n c", b=B, t=T, n=N)
        return x_ds, gx
    
    def pool_mv_aux(self, aux: Tensor):
        if aux is not None:
            B, T, N, C, H, W = aux.shape
            aux = rearrange(aux, "b t n c h w -> (b t n) c h w")
            aux_ds = self.frozen.encode_aux(aux)
            aux_ds = rearrange(aux_ds, "(b t n) l c -> b t n l c", b=B, t=T, n=N)
        else:
            aux_ds = None
        return aux_ds
    
    def ray_encoding(self, norm_xy: Tensor, extrinsic: Tensor):
        norm_xy_ds = self.pool_mv_aux(norm_xy)
        pe = plucker_ray_pe(norm_xy_ds, extrinsic)
        return norm_xy_ds, pe
    
    def forward(
        self, 
        rgb: Tensor, 
        norm_xy: Tensor, 
        extrinsic: Tensor, 
        **aux_tensors: Tensor
    ):
        """
        Args:
            rgb (Tensor): (B, T, N, 3, H, W)
            norm_xy (Tensor): (B, T, N, 2, H, W)
            extrinsic (Tensor): (B, T, N, 4, 4), ^ref_cam T
            aux_tensors (Tensor): each of (B, T, N, C, H, W)

        Returns:
            obs (Dict[str, Tensor]): image observations
                - x:    tensor of shape (B, To, Ncam, L, C) patch feature
                - gx:   tensor of shape (B, To, Ncam, C), projected global feature, not masked
                - pe:   tensor of shape (B, To, Ncam, L, 6), ray pe
                - aux:  {aux_user_key: tensor of shape (B, To, Ncam, L, ...)}
        """
        x_ds, gx = self.encode_mv_images(rgb)
        norm_xy_ds, pe = self.ray_encoding(norm_xy, extrinsic)
        aux_ds = {k:self.pool_mv_aux(a) for k, a in aux_tensors.items()}

        return {
            "x": x_ds,       # (B, T, N, L, C)
            "gx": gx,        # (B, T, N, C)
            "pe": pe,        # (B, T, N, L, 6)
            "aux": aux_ds,   # (B, T, N, L, ...) of each entry
        }


def plucker_ray_pe(norm_xy: Tensor, extrinsic: Tensor):
    """
    Args:
        norm_xy (Tensor): (..., L, 2)
        extrinsic (Tensor): (..., 4, 4), ^w_c T
    
    Returns:
        pe (Tensor): (..., L, 6)
    """
    homo_dir = F.pad(norm_xy, pad=(0, 1), mode="constant", value=1.0)
    direction = F.normalize(homo_dir, dim=-1)  # (..., L, 3)
    rotmat = extrinsic[..., :3, :3]  # (..., 3, 3)
    direction = direction @ rotmat.transpose(-1, -2)  # (..., L, 3), to world rays
    # (..., 3) -> (..., 1, 3) -> (..., L, 3)
    origin = extrinsic[..., :3, 3].unsqueeze(-2).expand_as(direction)  # (..., L, 3)
    pe = torch.cat([direction, torch.cross(direction, origin, dim=-1)], dim=-1)
    return pe

