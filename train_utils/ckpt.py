"""Checkpoint loading helpers.

The released checkpoints store only the action expert (`model.actor`); the VLM is
frozen pretrained weights pulled from HuggingFace at construction time, so it is never
serialized. See `Trainer.save_model`, which writes `model.actor.state_dict()`.

The point of this module is to fail *loudly and specifically* on a mismatch. A silent
partial load is the worst outcome: training proceeds, the loss curve looks plausible,
and you only find out at evaluation time that half the network was random.
"""

from typing import Dict, Optional
from torch import nn, Tensor


class CkptCompatReport(object):
    """Result of matching a checkpoint's tensors against a live module."""

    def __init__(self, matched, missing, unexpected, mismatched):
        self.matched = matched          # List[str], loaded successfully
        self.missing = missing          # List[str], in model but not in ckpt
        self.unexpected = unexpected    # List[str], in ckpt but not in model
        self.mismatched = mismatched    # List[(name, ckpt_shape, model_shape)]

    @property
    def is_exact(self):
        return not (self.missing or self.unexpected or self.mismatched)

    def summary(self, max_show: int = 8):
        lines = ["matched {} tensors".format(len(self.matched))]
        for label, items in (("missing (model has, ckpt lacks)", self.missing),
                             ("unexpected (ckpt has, model lacks)", self.unexpected)):
            if items:
                lines.append("  {}: {}".format(label, len(items)))
                lines += ["      {}".format(n) for n in items[:max_show]]
                if len(items) > max_show:
                    lines.append("      ... and {} more".format(len(items) - max_show))
        if self.mismatched:
            lines.append("  shape mismatch: {}".format(len(self.mismatched)))
            for name, cs, ms in self.mismatched[:max_show]:
                lines.append("      {}: ckpt {} vs model {}".format(name, cs, ms))
            if len(self.mismatched) > max_show:
                lines.append("      ... and {} more".format(len(self.mismatched) - max_show))
        return "\n".join(lines)


def _strip_prefix(weights: Dict[str, Tensor], prefix: str):
    """Tolerate checkpoints saved from the full VLA (keys prefixed with 'actor.')."""
    if weights and all(k.startswith(prefix) for k in weights):
        return {k[len(prefix):]: v for k, v in weights.items()}
    return weights


def inspect_actor_weights(actor: nn.Module, weights: Dict[str, Tensor]):
    """Compare a checkpoint's tensors against `actor` without mutating anything."""
    weights = _strip_prefix(weights, "actor.")
    model_sd = actor.state_dict()

    matched, mismatched = [], []
    for name, tensor in weights.items():
        if name not in model_sd:
            continue
        if tuple(tensor.shape) != tuple(model_sd[name].shape):
            mismatched.append((name, tuple(tensor.shape), tuple(model_sd[name].shape)))
        else:
            matched.append(name)

    missing = sorted(set(model_sd) - set(weights))
    unexpected = sorted(set(weights) - set(model_sd))
    return CkptCompatReport(sorted(matched), missing, unexpected, mismatched)


def load_actor_weights(
    actor: nn.Module,
    weights: Dict[str, Tensor],
    strict: bool = True,
    what: str = "checkpoint",
):
    """Load pretrained action-expert weights, reporting exactly what happened.

    Args:
        actor: the `ActionExpert` to load into
        weights: the `"weights"` entry of a checkpoint
        strict: if True (default) any mismatch raises. Set False only when you
            deliberately want a partial load -- e.g. loading a checkpoint from before
            upstream's `compact model` commit, which shrank `context_encoder.proj_*`
            and `context_encoder.post_attn`'s FFN. Those layers then stay randomly
            initialised and must be trained.
        what: label used in messages

    Returns:
        CkptCompatReport
    """
    report = inspect_actor_weights(actor, weights)

    if report.is_exact:
        print("[INFO] {} matches the model exactly ({} tensors)."
              .format(what, len(report.matched)))
        actor.load_state_dict(_strip_prefix(weights, "actor."))
        return report

    message = "{} does not match the model:\n{}".format(what, report.summary())

    if strict:
        raise RuntimeError(
            message + "\n\n"
            "If this checkpoint predates upstream's 'compact model' commit (7eb18ac),\n"
            "its ContextEncoder is wider (122.4M vs 102.3M params). Either use a\n"
            "checkpoint trained with the current model, or pass\n"
            "`--pretrained_strict False` to load the compatible subset and train the\n"
            "rest from scratch."
        )

    print("[WARN] " + message)
    print("[WARN] Loading the compatible subset only. The tensors listed above keep "
          "their random initialisation and WILL be trained from scratch.")
    filtered = {k: v for k, v in _strip_prefix(weights, "actor.").items()
                if k in report.matched}
    actor.load_state_dict(filtered, strict=False)
    return report
