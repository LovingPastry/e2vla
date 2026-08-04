import torch
import contextlib
from typing import Optional
from torch import Tensor, nn
import torch.nn.functional as F


def maybe_no_grad(module: nn.Module):
    """`torch.no_grad()` for a fully frozen `module`, a no-op context otherwise.

    The vision/text backbones are frozen, so their forwards are wrapped in `no_grad` and
    their activations are never kept. That wrapper must not stay unconditional: injecting
    LoRA (see `train_utils/lora.py:setup_vlm_lora`) makes part of a backbone trainable,
    and a `no_grad` above the factors detaches them *silently* -- the parameters would
    still sit in the optimizer and in the checkpoint, receive no gradient, and never
    move. Nothing fails; the run just trains the action expert alone.

    `nullcontext` rather than `enable_grad`: inference runs inside `torch.inference_mode`
    and forcing grad on there would rebuild the autograd graph for every denoising step.
    """
    for p in module.parameters():
        if p.requires_grad:
            return contextlib.nullcontext()
    return torch.no_grad()


def concat_mask(
    mask0: Optional[Tensor], 
    mask1: Optional[Tensor],
    L0: int, L1: int
):
    if mask0 is None and mask1 is None:
        return None
    
    if mask0 is not None:
        assert L0 == mask0.shape[1]
        assert mask0.dtype in (torch.bool, torch.float32)
    
    if mask1 is not None:
        assert L1 == mask1.shape[1]
        assert mask1.dtype in (torch.bool, torch.float32)
    
    if (mask0 is not None) and (mask1 is None):
        if mask0.dtype == torch.bool:
            mask = F.pad(mask0, (0, L1), value=True)
        elif mask0.dtype == torch.float32:
            mask = F.pad(mask0, (0, L1), value=0.0)
        return mask

    if (mask0 is None) and (mask1 is not None):
        if mask1.dtype == torch.bool:
            mask = F.pad(mask1, (L0, 0), value=True)
        elif mask1.dtype == torch.float32:
            mask = F.pad(mask1, (L0, 0), value=0.0)
        return mask
    
    # both are not None
    if mask0.dtype == torch.bool and mask1.dtype == torch.float32:
        mask0 = mask0.float().log()
    if mask0.dtype == torch.float32 and mask1.dtype == torch.bool:
        mask1 = mask1.float().log()
    mask = torch.cat([mask0, mask1], dim=-1)
    return mask


def simple_mlp(neurons: list, ln: bool = False):
    assert len(neurons) >= 2
    modules = [nn.Linear(neurons[0], neurons[1])]
    for i in range(1, len(neurons)-1):
        if ln:
            modules.append(nn.LayerNorm(neurons[i]))
        modules.append(nn.GELU())
        modules.append(nn.Linear(neurons[i], neurons[i+1]))
    return nn.Sequential(*modules)

