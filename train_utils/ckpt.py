"""Checkpoint loading helpers.

The released checkpoints store only the action expert (`model.actor`); the VLM is
frozen pretrained weights pulled from HuggingFace at construction time, so it is never
serialized. See `Trainer.save_model`, which writes `model.actor.state_dict()`.

The point of this module is to fail *loudly and specifically* on a mismatch. A silent
partial load is the worst outcome: training proceeds, the loss curve looks plausible,
and you only find out at evaluation time that half the network was random.

SCOPE, and it is narrow: these helpers compare tensor *names and shapes*. They cannot
see what the weights mean, so their success messages say the layout matched, never that
the checkpoint is the right one. Anything semantic has to be carried out of band --
which is what `OBJECTIVE` and `check_objective` below are for.
"""

from typing import Dict
from torch import nn, Tensor


# The generative objective the action head implements: DDIM epsilon prediction.
#
# `Trainer.save_model` stamps this into every checkpoint and `check_objective` verifies
# it on the way back in. That looks redundant while there is only one possible value,
# and it is -- today. The point is the failure it forecloses: a head trained on some
# other objective (a flow-matching variant was prototyped and rolled back, and could
# come back) has the *same parameter tensors*, so its checkpoint would load here with
# zero missing keys, report a clean layout match, and then sample nonsense. Names and
# shapes cannot distinguish the two; only this stamp can.
#
# The released pretrain checkpoints predate the stamp and carry no key. They are all
# DDIM, hence the default below.
OBJECTIVE = "ddim"


def check_objective(ckpt: dict, what: str = "checkpoint"):
    """Verify a checkpoint was trained with the objective this code implements.

    Args:
        ckpt: the loaded checkpoint dict
        what: label used in messages

    Returns:
        str, the checkpoint's objective (always `OBJECTIVE` if this returns at all)
    """
    objective = ckpt.get("objective", OBJECTIVE)
    if objective != OBJECTIVE:
        raise ValueError(
            "objective mismatch: the {} was trained with '{}', but this code implements "
            "'{}'. The weights would load without a single missing key -- the two heads "
            "have identical state_dict layouts -- and then sample nonsense, so this is "
            "refused rather than warned about."
            .format(what, objective, OBJECTIVE))
    return objective


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
        # "layout", not "matches": every tensor lined up by name and shape, which is all
        # this function can check. See the module docstring.
        print("[INFO] {}: state_dict layout matches ({} tensors, names and shapes)."
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
