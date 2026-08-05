"""TensorBoard plumbing for `train.py`.

The training loop only needs `TBLogger.scalars(...)`; everything that has to walk the
parameter tree to produce a number lives here, so the loop keeps reading like a loop.

What is worth logging, and why each one exists -- a loss curve alone cannot distinguish
any of these failures:

* per-module gradient norms (`gnorm/*`)
      "the loss stopped moving" has completely different fixes depending on whether the
      gradient died in the ContextEncoder, in the DiffusionHead, or in the VLM LoRA
      factors. One curve per chunk of the model tells you which.
* AdamW's actual update size (`optim/update_norm`, `optim/update_ratio`)
      the grad norm is not the step size -- Adam normalises it away. What matters for
      picking `max_lr` is ||update|| / ||params||, which should sit around 1e-3; an order
      of magnitude either side is a badly chosen LR and is invisible in the loss for
      thousands of iterations.
* dataloader wait (`perf/data_wait_ms`)
      this repo reads HDF5 through `workers` processes and decodes video frames. When the
      loader starves, step time doubles with no other symptom -- and the fix (`workers`,
      `sample_multiplex`) is nothing to do with the model.

Nothing here is on the hot path: every function is called once per `log_interval`.
"""

import os
from typing import Dict, List, Optional

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter


def param_group_name(name: str) -> str:
    """Fully-qualified parameter name -> the chunk of `VLA` it belongs to.

    Matches the module boundaries that are actually independent knobs: the two halves of
    the action expert are trained together but fail separately, and `vlm.*` only ever has
    gradients when VLM LoRA is on (the backbones themselves are frozen).
    """
    if ".conv_tower." in name:
        return "conv_tower"
    if name.startswith("vlm."):
        return "vlm_lora"
    if name.startswith("actor.context_encoder."):
        return "context_encoder"
    if name.startswith("actor.dp_head."):
        return "dp_head"
    return "other"


@torch.no_grad()
def grad_norms(model: nn.Module) -> Dict[str, float]:
    """L2 norm of the gradient, total and per module group.

    Call this AFTER `scaler.unscale_` (or with AMP off) -- a scaled gradient's norm is off
    by the scaler's current factor, which itself moves around, so the curve would be
    unreadable. Grads survive `optimizer.step()`, so calling it just before the next
    `zero_grad()` is fine.
    """
    sums: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        sq = p.grad.detach().float().pow(2).sum()
        key = param_group_name(name)
        sums[key] = sq if key not in sums else sums[key] + sq

    if not sums:
        return {}
    keys = sorted(sums)
    stacked = torch.stack([sums[k] for k in keys])
    values = torch.cat([stacked, stacked.sum(dim=0, keepdim=True)]).sqrt().tolist()
    out = {"gnorm/{}".format(k): v for k, v in zip(keys, values[:-1])}
    out["gnorm/total"] = values[-1]
    return out


@torch.no_grad()
def param_norm(model: nn.Module) -> float:
    """L2 norm over all trainable parameters. Only useful as the denominator of
    `optim/update_ratio`, but cheap enough to keep as its own curve -- a weight norm that
    grows without bound is the visible half of a weight-decay/LR mismatch."""
    total = None
    for p in model.parameters():
        if not p.requires_grad:
            continue
        sq = p.detach().float().pow(2).sum()
        total = sq if total is None else total + sq
    return 0.0 if total is None else float(total.sqrt())


@torch.no_grad()
def adamw_update_norm(optimizer: optim.Optimizer) -> float:
    """Norm of the parameter delta AdamW is about to apply, reconstructed from its state.

    Reconstructed rather than measured, because measuring it means keeping a full copy of
    the weights around (400MB at `base`) just to subtract. The formula is AdamW's update
    verbatim -- bias-corrected first moment over the square root of the bias-corrected
    second, plus the decoupled decay term -- so it agrees with the real step to floating
    point, and it costs a few elementwise passes once per log interval.

    Returns 0.0 before the first step, when the optimizer has no state yet.
    """
    total = None
    for group in optimizer.param_groups:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        wd = group["weight_decay"]
        for p in group["params"]:
            state = optimizer.state.get(p, None)
            if not state or "exp_avg" not in state:
                continue
            step = state["step"]
            step = float(step.item() if torch.is_tensor(step) else step)
            if step <= 0:
                continue
            bias1 = 1.0 - beta1 ** step
            bias2 = 1.0 - beta2 ** step
            m = state["exp_avg"].float() / bias1
            v = state["exp_avg_sq"].float() / bias2
            upd = lr * m / (v.sqrt() + eps)
            if wd != 0:
                upd = upd + lr * wd * p.detach().float()
            sq = upd.pow(2).sum()
            total = sq if total is None else total + sq
    return 0.0 if total is None else float(total.sqrt())


def trainable_param_table(model: nn.Module) -> str:
    """Markdown table of trainable / total parameters per module group, for `add_text`.

    Worth having in the run itself rather than only on stdout: which parameters were
    actually free to move is the first thing you want when comparing two runs months
    later, and it is exactly what LoRA and the frozen backbones make non-obvious.
    """
    stats: Dict[str, List[int]] = {}
    for name, p in model.named_parameters():
        key = param_group_name(name)
        row = stats.setdefault(key, [0, 0])
        row[0] += p.numel() if p.requires_grad else 0
        row[1] += p.numel()

    lines = ["| module | trainable | total |", "| --- | ---: | ---: |"]
    for key in sorted(stats):
        train_n, total_n = stats[key]
        lines.append("| {} | {:.3f}M | {:.3f}M |"
                     .format(key, train_n / 1e6, total_n / 1e6))
    train_n = sum(v[0] for v in stats.values())
    total_n = sum(v[1] for v in stats.values())
    lines.append("| **all** | **{:.3f}M** | **{:.3f}M** |"
                 .format(train_n / 1e6, total_n / 1e6))
    return "\n".join(lines)


class TBLogger(object):
    """A SummaryWriter that is a no-op when the run is not being saved.

    `train.py` may run without `-s`, in which case nothing should be written but the loop
    should not have to branch on it at every call site.
    """

    def __init__(self, log_dir: Optional[str]):
        self.log_dir = log_dir
        self.writer: Optional[SummaryWriter] = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            # 30s instead of the 120s default: the point of these curves is to catch a
            # diverging run early, which is not helped by a two-minute delay.
            self.writer = SummaryWriter(log_dir, flush_secs=30)

    @property
    def enabled(self) -> bool:
        return self.writer is not None

    def scalars(self, values: Dict[str, float], step: int):
        if self.writer is None:
            return
        for key, val in values.items():
            if val is None:
                continue
            self.writer.add_scalar(key, val, step)

    def text(self, tag: str, body: str, step: int = 0):
        if self.writer is None:
            return
        self.writer.add_text(tag, body, step)

    def image_grid(self, tag: str, images: torch.Tensor, step: int, gap: int = 2):
        """Tile `images` (N, 3, H, W) in [0, 1] side by side, left to right.

        Written out rather than delegated to `torchvision.utils.make_grid`: a handful of
        camera views is one row, and this file then stays importable without pulling in
        torchvision, whose import order relative to `envars` is load-bearing everywhere
        else in this repo.
        """
        if self.writer is None:
            return
        imgs = images.detach().float().clamp(0, 1).cpu()
        if imgs.ndim == 3:
            imgs = imgs[None]
        if imgs.shape[0] > 1 and gap > 0:
            sep = imgs.new_ones(imgs.shape[0] - 1, imgs.shape[1], imgs.shape[2], gap)
            tiles = [t for pair in zip(imgs[:-1], sep) for t in pair] + [imgs[-1]]
        else:
            tiles = list(imgs)
        self.writer.add_image(tag, torch.cat(tiles, dim=-1), step)

    @torch.no_grad()
    def histograms(self, model: nn.Module, step: int, max_elems: int = 65536):
        """Weight and gradient distributions per trainable tensor.

        Strided-subsampled to `max_elems`: a faithful histogram of a 60M-parameter module
        costs more than the training step that produced it, and the shape of the
        distribution -- which is all this is for -- survives the subsample.
        """
        if self.writer is None:
            return
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            flat = p.detach().flatten()
            stride = max(1, flat.numel() // max_elems)
            self.writer.add_histogram("weights/" + name, flat[::stride].float(), step)
            if p.grad is not None:
                gflat = p.grad.detach().flatten()
                self.writer.add_histogram("grads/" + name, gflat[::stride].float(), step)

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
