from typing import List, Optional
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

    `use_language=False` turns this into a pure vision encoder for the vision-action
    variants (`TrainConfig.context_encoder` != "vl"). The SigLIP text tower is not merely
    left unused, it is *deleted*: it is ~110M frozen parameters that would otherwise sit
    on the GPU for the whole run, and deleting it makes any accidental language path fail
    with an AttributeError instead of quietly re-appearing. Nothing is lost from the
    checkpoint either way -- `train.py` only ever serializes `model.actor`.
    """

    def __init__(
        self,
        use_language: bool = True,
    ):
        super().__init__()

        self.use_language = use_language
        self.siglip = Siglip()
        self.dinov2 = Dinov2()
        if not use_language:
            self._drop_text_tower()
        for p in self.siglip.parameters():
            p.requires_grad_(False)
        for p in self.dinov2.parameters():
            p.requires_grad_(False)

    def _drop_text_tower(self):
        """Remove SigLIP's text half. Only the vision tower is ever called after this.

        `SiglipModel` holds the two towers as plain submodules, so deleting the attribute
        is enough -- the vision path never touches `text_model`, and the contrastive
        `match()` helper (which does) is not part of this repo's forward. A
        `vlm_lora_targets` entry naming "siglip_text" then raises in `setup_vlm_lora`,
        which is the right outcome: there is nothing to adapt.
        """
        siglip_model = self.siglip.frozen.siglip.siglip
        num = sum(p.numel() for p in siglip_model.text_model.parameters())
        del siglip_model.text_model
        print("[INFO] VLM: language disabled, SigLIP text tower dropped ({:.1f}M frozen "
              "parameters not allocated)".format(num / 1e6))

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
        obs_norm_xys: Optional[Tensor],
        obs_extrinsics: Optional[Tensor],
        prompt_text: Optional[List[str]],
        fp16: bool
    ):
        """
        `obs_norm_xys` / `obs_extrinsics` / `prompt_text` are all optional, and None means
        the corresponding modality is genuinely absent rather than merely unused: `VLA`
        nulls them out for the vision-action context encoders so nothing downstream can
        reach a camera parameter or a token of language by accident.
        """
        # Each of the three calls re-scopes grad to its own tower (see the encoders), so
        # this outer context only has to get out of the way when *something* is trainable.
        # It cannot be dropped: a plain `no_grad` here would override them all.
        with maybe_no_grad(self):
            x_dinov2, gx_dinov2 = self.dinov2.encode_mv_images(rgbs)
            x_siglip, gx_siglip = self.siglip.encode_mv_images(rgbs)
            if self.use_language:
                if prompt_text is None:
                    raise ValueError(
                        "VLM was built with use_language=True but got prompt_text=None. "
                        "Pass the instruction, or build the model with a vision-action "
                        "context_encoder.")
                x_text, gx_text = self.siglip.encode_text(prompt_text)

        # `pool_mv_aux` is None-safe: without intrinsics there are no normalized-plane
        # coordinates to downsample, and the context encoder falls back to a fixed grid.
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
            "norm_xy_ds": None if norm_xy_ds is None else norm_xy_ds[:, -1],
            "vision_embeds": [x_dinov2[:, -1],
                              x_siglip[:, -1]],     # List of (B, Ncam, Lv, C)
            "lang_embeds": [x_text] if self.use_language else None,
            "lang_mask": None,                      # (B, La)
            "extrinsics": None if obs_extrinsics is None else obs_extrinsics[:, -1],
        }
        # NOTE: xxx[:, -1] selects the latest image observation. We don't use history images

        return obs, feature
