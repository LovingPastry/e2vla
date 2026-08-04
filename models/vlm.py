from typing import List
from torch import nn, Tensor
from .encoders.dino import Encoder as Dinov2
from .encoders.siglip import Encoder as Siglip
from .layers.utils import maybe_no_grad


class VLM(nn.Module):
    """Frozen SigLIP (image + text) and DINOv2, run once per step to produce tokens.

    "Frozen" is the default, not a property of the class: `TrainConfig.vlm_lora_rank`
    can inject LoRA into either backbone (`train_utils/lora.py:setup_vlm_lora`), which
    leaves every pretrained weight here frozen and trains only the added factors. Every
    `no_grad` in this file and in the two encoders is therefore conditional -- see
    `maybe_no_grad`. Nothing else about this module changes.
    """

    def __init__(
        self,
    ):
        super().__init__()

        self.siglip = Siglip()
        self.dinov2 = Dinov2()
        for p in self.siglip.parameters():
            p.requires_grad_(False)
        for p in self.dinov2.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        """Keep both backbones in eval mode even while the policy trains.

        `Dinov2Encoder.__init__` calls `.eval()` on itself and `Trainer.fitting()`'s
        `model.train()` silently undid it. That was harmless as long as nothing here had
        gradients -- neither backbone has BatchNorm and both ship with dropout at 0 --
        but it stops being a detail once LoRA makes them part of the optimisation, where
        the adapted features would start depending on whatever regularisation the
        upstream config happens to enable. The LoRA factors themselves have no
        train/eval behaviour, so nothing is lost by pinning this.
        """
        super().train(mode)
        self.siglip.eval()
        self.dinov2.eval()
        return self

    def forward(
        self,
        rgbs: Tensor,
        obs_norm_xys: Tensor,
        obs_extrinsics: Tensor,
        prompt_text: List[str],
        fp16: bool
    ):

        # Each of the three calls re-scopes grad to its own tower (see the encoders), so
        # this outer context only has to get out of the way when *something* is trainable.
        # It cannot be dropped: a plain `no_grad` here would override them all.
        with maybe_no_grad(self):
            x_dinov2, gx_dinov2 = self.dinov2.encode_mv_images(rgbs)
            x_siglip, gx_siglip = self.siglip.encode_mv_images(rgbs)
            x_text, gx_text = self.siglip.encode_text(prompt_text)

        norm_xy_ds = self.siglip.pool_mv_aux(obs_norm_xys)

        obs = {
            "rgb": rgbs,
            "norm_xy": obs_norm_xys,
            "extrinsics": obs_extrinsics,
            "text": prompt_text,
        }

        # Every patch of every camera is valid: images are dense and the dataset no
        # longer carries segmentation, so attention runs unmasked over vision tokens.
        feature = {
            "norm_xy_ds": norm_xy_ds[:, -1],        # (B, Ncam, Lv, 2)
            "vision_embeds": [x_dinov2[:, -1],
                              x_siglip[:, -1]],     # List of (B, Ncam, Lv, C)
            "lang_embeds": [x_text],                # List of (B, La, C)
            "lang_mask": None,                      # (B, La)
            "extrinsics": obs_extrinsics[:, -1]
        }
        # NOTE: xxx[:, -1] selects the latest image observation. We don't use history images

        return obs, feature

