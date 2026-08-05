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
which is what `check_objective` below (and its siblings for the action space, the action
normalization and the VLM's LoRA) are for.
"""

from typing import Dict, Sequence
from torch import nn, Tensor


def check_objective(ckpt: dict, objective: str, what: str = "checkpoint"):
    """Verify a checkpoint was trained with the generative objective this run uses.

    The action head supports two (see `models/action_expert.py:OBJECTIVES`): DDIM epsilon
    prediction and rectified-flow matching. They share *every parameter tensor*, name and
    shape alike -- the network is the same conditional vector field, only the meaning of
    its output changes. So a flow checkpoint loads into a DDIM model without a single
    missing key, reports a clean layout match, and then hands a velocity field to a
    noise-prediction sampler. Nothing but this stamp can tell them apart, which is why the
    mismatch is refused rather than warned about.

    `Trainer.save_model` writes `cfg.objective` into every checkpoint it produces. The
    released pretrain checkpoints predate the stamp and carry no key; they are all DDIM,
    hence the default below.

    Args:
        ckpt: the loaded checkpoint dict
        objective: the objective this run is configured for
        what: label used in messages

    Returns:
        str, the checkpoint's objective (always `objective` if this returns at all)
    """
    from models.action_expert import DEFAULT_OBJECTIVE
    stored = ckpt.get("objective", DEFAULT_OBJECTIVE)
    if stored != objective:
        raise ValueError(
            "objective mismatch: the {} was trained with '{}', but this run is configured "
            "for '{}'. The weights would load without a single missing key -- the two "
            "heads have identical state_dict layouts -- and then sample nonsense, so this "
            "is refused rather than warned about.\n\n"
            "Set `objective` in the config to match the checkpoint, or start from a "
            "checkpoint trained with the objective you want. There is no conversion: the "
            "two are different regression problems on the same weights."
            .format(what, stored, objective))
    return stored


def check_action_layout(ckpt: dict, layout: str, what: str = "checkpoint"):
    """Verify a checkpoint predicts in the action space this run is configured for.

    Same failure mode as `check_objective`, one axis over: `models/action_space.py`
    offers an EE-pose space (10 channels) and a joint space (nq+1 channels). Those
    usually differ in width, and then the state_dict layout check catches the mismatch on
    its own. They do not always -- a 9-joint arm also lands on 10 -- and in that case
    every tensor lines up by name and shape, the load reports a clean match, and the head
    then denoises joint angles as if they were metres and a rotation.

    Checkpoints written before this stamp existed are all EE-pose, hence the default.

    Args:
        ckpt: the loaded checkpoint dict
        layout: `ActionSpace.layout` for the current run
        what: label used in messages

    Returns:
        str, the checkpoint's action layout
    """
    from models.action_norm import ACTION_LAYOUT
    stored = ckpt.get("action_layout", ACTION_LAYOUT)
    if stored != layout:
        raise ValueError(
            "action space mismatch: the {} predicts in layout '{}', but this run is "
            "configured for '{}'. Set `action_space` in the config to match, or start "
            "from a checkpoint trained in the right space -- the two encodings can share "
            "a channel count, in which case nothing downstream would notice."
            .format(what, stored, layout))
    return stored


def check_action_norm(ckpt: dict, action_norm, what: str = "checkpoint",
                      strict: bool = True):
    """Verify a checkpoint's action normalization matches the one being used.

    Same class of hazard as `check_objective`, and it is worth spelling out because the
    failure is even quieter. q01/q99 normalization is an affine reparameterisation of
    the action space; it changes *no* tensor name and *no* tensor shape. A checkpoint
    trained on normalized actions, loaded into a model with no normalizer, loads
    perfectly, trains to a plausible loss, and drives the arm with actions that are off
    by roughly the inverse of the affine -- a factor of ~20 on translation. Nothing but
    the stored statistics can catch it.

    Args:
        ckpt: the loaded checkpoint dict
        action_norm: `ActionNormalizer` or None, what this run is about to use
        what: label used in messages
        strict: raise on mismatch. False downgrades to a warning, which is what
            fine-tuning wants: adapting a checkpoint to a new action normalization is a
            legitimate thing to do, it just must not happen by accident.

    Returns:
        bool, True if the checkpoint's normalization matches `action_norm`
    """
    # A checkpoint predating this feature carries no key, which means "no normalization"
    # -- the same default the config has, so the common case stays silent.
    stored = ckpt.get("action_norm", None)

    if stored is None and action_norm is None:
        return True

    if stored is not None and action_norm is not None:
        from models.action_norm import ActionNormalizer
        # Read the stored stats under THEIR own layout, not the current run's. This is a
        # comparison, and a cross-space comparison must be able to return "different"
        # rather than raise: `matches` already treats a layout difference as a mismatch,
        # which routes into the message below. Passing the run's layout here would make
        # resuming any non-default-space run raise inside its own consistency check.
        stored_layout = stored.get("layout", "cam_rel_t3r6_openness")
        if action_norm.matches(ActionNormalizer.from_dict(
                stored, what=what, expect_layout=stored_layout)):
            return True
        message = ("{}: action normalization statistics differ from the ones this run "
                   "uses. The weights load without a single missing key -- normalization "
                   "is an affine on the action space, not a tensor -- and the policy then "
                   "acts on a mis-scaled action.".format(what))
    elif stored is None:
        message = ("{}: was trained WITHOUT action normalization, but this run uses "
                   "q01/q99 normalized actions. The action space the weights were fit to "
                   "is not the one they are about to be used in.".format(what))
    else:
        message = ("{}: was trained WITH q01/q99 action normalization, but this run has "
                   "no action stats (action_norm_stats is unset). Predictions would be "
                   "interpreted in normalized units and executed as metres."
                   .format(what))

    if strict:
        raise ValueError(
            message + "\n\nEither point --action_norm_stats at the statistics this "
            "checkpoint was trained with, or pass the flag that relaxes this check if "
            "you are deliberately re-normalizing during fine-tuning.")

    print("[WARN] " + message)
    print("[WARN] Continuing anyway. This is only correct if you intend to re-fit the "
          "action head to the new action space -- expect the first few thousand steps to "
          "look like training from scratch on the affected layers.")
    return False


def read_vlm_lora_spec(ckpt: dict):
    """What VLM-side LoRA a checkpoint was trained with, as (rank, targets).

    Everything written before this feature existed has fully frozen backbones, and rank
    0 says exactly that -- so the absent key needs no special case anywhere upstream.
    """
    rank = int(ckpt.get("vlm_lora_rank", 0))
    targets = list(ckpt.get("vlm_lora_targets", []))
    return rank, targets


def check_vlm_lora(ckpt: dict, rank: int, targets, what: str = "checkpoint",
                   strict: bool = True):
    """Verify a checkpoint's VLM-side LoRA matches what this run is configured for.

    The failure this prevents is quiet in the usual way: the VLM's weights are not in the
    checkpoint at all (they are pulled from HuggingFace/torch.hub at construction), so a
    checkpoint whose backbones were adapted and a run that does not re-inject that LoRA
    differ by nothing a state_dict check can see. The action expert loads perfectly and
    then reads features from a backbone it was never trained against.

    Args:
        ckpt: the loaded checkpoint dict
        rank: `vlm_lora_rank` for the current run
        targets: `vlm_lora_targets` for the current run
        what: label used in messages
        strict: raise on mismatch, else warn. Fine-tuning a run that had no VLM LoRA into
            one that does is legitimate (the factors just start from zero); the reverse --
            dropping factors the checkpoint was fitted with -- is a silent regression, so
            it warns rather than passing unremarked.

    Returns:
        bool, True if the checkpoint's VLM LoRA matches this run's
    """
    ckpt_rank, ckpt_targets = read_vlm_lora_spec(ckpt)
    if ckpt_rank == rank and (ckpt_rank == 0 or sorted(ckpt_targets) == sorted(targets)):
        return True

    message = ("{}: VLM LoRA differs -- checkpoint has rank {} on {}, this run is "
               "configured for rank {} on {}. The backbones are not stored in the "
               "checkpoint, so nothing downstream can detect this: the action expert "
               "would load cleanly and then read features from a differently-adapted "
               "encoder."
               .format(what, ckpt_rank, ckpt_targets or "nothing", rank,
                       targets if rank else "nothing"))

    if strict:
        raise ValueError(message + "\n\nSet vlm_lora_rank / vlm_lora_targets in the "
                         "config to match the checkpoint.")

    if ckpt_rank > 0:
        print("[WARN] " + message)
        print("[WARN] The checkpoint's LoRA factors are DISCARDED. Its action expert was "
              "fitted to the adapted features and now sees the raw ones.")
    else:
        print("[INFO] {}: pretrained checkpoint has fully frozen backbones; this run adds "
              "rank-{} LoRA on {}. The factors start from zero (injection is a no-op at "
              "init), so this is a clean warm start.".format(what, rank, targets))
    return False


def load_vlm_lora_weights(vlm, weights: Dict[str, Tensor], what: str = "checkpoint"):
    """Restore the LoRA factors of an already-injected VLM.

    Exact match required in both directions, and deliberately so: this is a small,
    fully-determined tensor set, and any disagreement means the injection that produced
    the checkpoint is not the injection that just ran.

    Args:
        vlm: the `VLM` module, already passed through `setup_vlm_lora`
        weights: the checkpoint's `"vlm_lora_weights"` entry
        what: label used in messages
    """
    expected = {name for name, p in vlm.named_parameters() if p.requires_grad}
    got = set(weights)

    if expected != got:
        raise RuntimeError(
            "{}: VLM LoRA weights do not match the injected factors.\n"
            "  missing (model has, ckpt lacks): {}\n"
            "  unexpected (ckpt has, model lacks): {}\n"
            "Inject with the same vlm_lora_rank / vlm_lora_targets the checkpoint was "
            "trained with."
            .format(what, sorted(expected - got)[:8], sorted(got - expected)[:8]))

    # strict=False because `weights` covers only the factors, never the frozen backbone
    vlm.load_state_dict(weights, strict=False)
    print("[INFO] {}: loaded {} VLM LoRA tensors.".format(what, len(weights)))


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
    allow_missing_prefixes: Sequence[str] = (),
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
        allow_missing_prefixes: key prefixes the model may legitimately have and the
            checkpoint legitimately lack -- a module this run added that no released
            checkpoint could contain, currently only the conv branch. The narrow
            alternative to `strict=False`: this permits *exactly* those tensors to start
            from their initialisation, while an unexpected key, a shape change, or a
            missing key anywhere else still raises. `strict=False` would wave all of them
            through together, which is a much bigger claim than the one being made here.

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

    new_modules = [n for n in report.missing
                   if any(n.startswith(p) for p in allow_missing_prefixes)]
    if (allow_missing_prefixes and not report.unexpected and not report.mismatched
            and len(new_modules) == len(report.missing)):
        print("[INFO] {}: {} tensors matched. {} tensors belong to modules this run adds "
              "and the checkpoint cannot contain; they keep their initialisation and are "
              "trained:".format(what, len(report.matched), len(new_modules)))
        for prefix in allow_missing_prefixes:
            n = sum(1 for k in new_modules if k.startswith(prefix))
            if n:
                print("           {}* ({} tensors)".format(prefix, n))
        actor.load_state_dict(
            {k: v for k, v in _strip_prefix(weights, "actor.").items()
             if k in report.matched},
            strict=False)
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
