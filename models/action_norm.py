"""Per-dimension q01/q99 action normalization, loaded from a JSON stats file.

WHAT IS NORMALIZED, and it is not the raw dataset states. The model's action space is
the 10-dim camera-relative encoding produced by `action_expert.space_ee2cam` plus the
rescaled gripper openness -- 3 translation + 6D rotation + 1 openness. Statistics
computed over raw world-frame poses would be meaningless here, so the stats file must
be produced by `data_prepare/compute_action_stats.py`, which runs the exact same
`states2action` path the model does.

WHY. The DDIM head samples from N(0, I) and predicts epsilon, which implicitly assumes
the clean data it is denoising has roughly unit scale. Today's action vector does not:
translation deltas are ~1e-2 metres while the 6D rotation of a near-identity delta sits
at [1, 0, 0, 0, 1, 0], i.e. two channels pinned near 1 and four near 0. Mapping each
channel's [q01, q99] to [-1, 1] puts every channel on the same footing.

WHY QUANTILES rather than mean/std: teleoperated demonstrations contain rare, large
jumps (dropped tracking, operator resets). A std computed over those is dominated by
the tail, which squashes the 98% of motion that matters. q01/q99 bound the useful range
and let the outliers clip.

The transform is a pure affine, so it is exactly invertible except where clipping is
requested:

    normalize:    y = 2 * (x - q01) / (q99 - q01) - 1
    unnormalize:  x = (y + 1) / 2 * (q99 - q01) + q01

Buffers are registered non-persistent on purpose: these are data statistics, not learned
parameters, and adding them to `state_dict` would make every released (pre-normalization)
checkpoint fail the strict layout check in `train_utils/ckpt.py`. They travel with a
checkpoint through the explicit `"action_norm"` entry instead -- see `Trainer.save_model`
and `train_utils.ckpt.check_action_norm`.
"""

import json
from typing import Optional, Sequence

import torch
from torch import nn, Tensor


# Bumped only on an incompatible change to the JSON layout below.
STATS_VERSION = 1

# The default action layout these statistics are defined over, i.e. the one the EE-pose
# action space uses. Stamped into the JSON and checked on load: a stats file computed for
# a different action encoding would apply silently.
#
# Since `models/action_space.py` introduced a second action space (absolute joint angles),
# the layout is no longer a constant of the code -- it is a property of the configured
# `ActionSpace`, whose `.layout` string is what actually gets stamped and checked. This
# name stays as the default so pre-existing stats files (which are all EE-pose) and every
# caller that does not care keep working unchanged.
ACTION_LAYOUT = "cam_rel_t3r6_openness"

# Channels whose q99-q01 is below this are treated as constant and left untouched
# (scale 1, offset 0) rather than blown up by a division by ~0. A dimension can be
# genuinely constant -- e.g. a gripper that never closes within the recorded window.
MIN_RANGE = 1e-6

# Which channels clipping applies to: the 3 translation components and openness, NOT
# the 6D rotation. This is not a tuning knob, it is a correctness constraint.
#
# Clipping is meaningful for metric quantities -- an out-of-range translation is a
# dangerous end-effector jump, an out-of-range openness is not a gripper command. The 6D
# rotation is neither: `rotation_6d_to_matrix` runs Gram-Schmidt on it, so it is a pair
# of directions read up to scale, and an out-of-range value there is harmless. Clamping
# it is actively wrong. Saturate all six channels against the same bound and the two
# 3-vectors come out parallel; Gram-Schmidt then returns a matrix with a zero row, which
# is not a rotation at all. That state is reachable from any model whose rotation output
# runs out of range -- an untrained one does it on the first forward pass -- and it
# propagates a singular pose downstream instead of raising.
DEFAULT_CLIP_DIMS = (0, 1, 2, 9)


class ActionNormalizer(nn.Module):
    """Affine map from the raw camera-relative action space to roughly [-1, 1].

    Args:
        q01: (D,) 1st percentile per action channel
        q99: (D,) 99th percentile per action channel
        clip: clamp to [-1, 1] in `normalize` and to the same range before the inverse
            in `unnormalize`. On the training side this bounds the 2% of samples outside
            the quantile range; on the inference side it stops an out-of-distribution
            sample from denormalizing into a large end-effector jump. Applies only to
            the channels in `clip_dims`.
        clip_dims: which channels `clip` covers; see `DEFAULT_CLIP_DIMS` for why this is
            not all of them. None selects the default for a 10-dim action, or every
            channel for any other width.
        meta: free-form provenance carried through the JSON round trip; ignored by the
            math, but it is what tells you months later which datasets these came from.
    """

    def __init__(
        self,
        q01: Sequence[float],
        q99: Sequence[float],
        clip: bool = True,
        clip_dims: Optional[Sequence[int]] = None,
        meta: Optional[dict] = None,
        layout: str = ACTION_LAYOUT,
    ):
        super().__init__()
        self.layout = layout
        q01 = torch.as_tensor(list(q01), dtype=torch.float32)
        q99 = torch.as_tensor(list(q99), dtype=torch.float32)
        if q01.shape != q99.shape or q01.ndim != 1:
            raise ValueError("q01 and q99 must be 1-D and the same length, got {} and {}"
                             .format(tuple(q01.shape), tuple(q99.shape)))
        if torch.any(q99 < q01):
            bad = torch.nonzero(q99 < q01).ravel().tolist()
            raise ValueError("q99 < q01 on channel(s) {} -- the stats file is corrupt"
                             .format(bad))

        span = q99 - q01
        degenerate = span < MIN_RANGE
        # scale/offset are stored, not q01/q99, so the hot path is one fused mul-add
        scale = torch.where(degenerate, torch.ones_like(span), 2.0 / span.clamp(min=MIN_RANGE))
        offset = torch.where(degenerate, torch.zeros_like(q01), q01)
        shift = torch.where(degenerate, torch.zeros_like(span), torch.ones_like(span))

        self.register_buffer("q01", q01, persistent=False)
        self.register_buffer("q99", q99, persistent=False)
        self.register_buffer("scale", scale, persistent=False)
        self.register_buffer("offset", offset, persistent=False)
        self.register_buffer("shift", shift, persistent=False)
        self.register_buffer("degenerate", degenerate, persistent=False)

        D = q01.numel()
        if clip_dims is None:
            # Keyed on the layout, not on D: the 6D-rotation carve-out below is a property
            # of the EE-pose encoding, and a joint space with 9 joints would also be
            # 10-dim while needing every channel clipped.
            clip_dims = (DEFAULT_CLIP_DIMS if (layout == ACTION_LAYOUT and D == 10)
                         else tuple(range(D)))
        clip_dims = tuple(int(i) for i in clip_dims)
        if any(i < 0 or i >= D for i in clip_dims):
            raise ValueError("clip_dims {} out of range for a {}-dim action"
                             .format(clip_dims, D))
        clip_mask = torch.zeros(D, dtype=torch.bool)
        clip_mask[list(clip_dims)] = True
        self.register_buffer("clip_mask", clip_mask, persistent=False)

        self.clip = bool(clip)
        self.clip_dims = clip_dims
        self.meta = dict(meta) if meta else {}

    @property
    def action_dim(self):
        return self.q01.numel()

    def _slices(self, dim: int):
        """Stats for the leading `dim` channels.

        Callers pass either the full 10-dim action or just its 9-dim t3r6 part (the
        geometry path in `DiffusionHead.forward` drops openness), and the two share a
        prefix by construction.
        """
        if dim > self.action_dim:
            raise ValueError("got a {}-dim action but the stats cover only {} channels"
                             .format(dim, self.action_dim))
        return (self.scale[:dim], self.offset[:dim], self.shift[:dim],
                self.clip_mask[:dim])

    def _clamp(self, y: Tensor, clip_mask: Tensor):
        if not self.clip:
            return y
        return torch.where(clip_mask, y.clamp(-1.0, 1.0), y)

    def normalize(self, x: Tensor):
        """Raw camera-relative action -> normalized. Args/Returns: (..., D<=action_dim)."""
        scale, offset, shift, clip_mask = self._slices(x.shape[-1])
        return self._clamp((x - offset) * scale - shift, clip_mask)

    def unnormalize(self, y: Tensor):
        """Normalized action -> raw camera-relative. Args/Returns: (..., D<=action_dim)."""
        scale, offset, shift, clip_mask = self._slices(y.shape[-1])
        y = self._clamp(y, clip_mask)
        return (y + shift) / scale + offset

    def extra_repr(self):
        return "action_dim={}, clip={}, clip_dims={}, degenerate_dims={}".format(
            self.action_dim, self.clip, self.clip_dims,
            torch.nonzero(self.degenerate).ravel().tolist())

    ############################ serialization ############################

    def to_dict(self):
        return {
            "version": STATS_VERSION,
            "layout": self.layout,
            "action_dim": self.action_dim,
            "clip": self.clip,
            "clip_dims": list(self.clip_dims),
            "q01": self.q01.tolist(),
            "q99": self.q99.tolist(),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict, what: str = "action stats",
                  expect_layout: str = ACTION_LAYOUT):
        """`expect_layout` is the configured `ActionSpace.layout`. It defaults to the
        EE-pose layout so pre-existing callers and stats files are unaffected."""
        version = d.get("version", STATS_VERSION)
        if version != STATS_VERSION:
            raise ValueError("{}: stats version {} but this code writes version {}"
                             .format(what, version, STATS_VERSION))
        # A file with no "layout" key predates the second action space, so it is EE-pose.
        # That default is only safe against ACTION_LAYOUT -- against any other expectation
        # an unstamped file is exactly the silent-misapplication case this guards.
        layout = d.get("layout", ACTION_LAYOUT)
        if layout != expect_layout:
            raise ValueError(
                "{}: statistics were computed for action layout '{}', but the model uses "
                "'{}'. The two encodings can have identical channel counts, so nothing "
                "downstream would notice. Recompute with "
                "data_prepare/compute_action_stats.py under the right --config."
                .format(what, layout, expect_layout))
        return cls(q01=d["q01"], q99=d["q99"], clip=d.get("clip", True),
                   clip_dims=d.get("clip_dims", None), meta=d.get("meta", {}),
                   layout=layout)

    @classmethod
    def from_json(cls, path: str, expect_layout: str = ACTION_LAYOUT):
        with open(path, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        return cls.from_dict(d, what=path, expect_layout=expect_layout)

    def to_json(self, path: str):
        import os
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, ensure_ascii=False, indent=4)

    ############################ comparison ############################

    def matches(self, other: Optional["ActionNormalizer"], atol: float = 1e-6):
        """True when `other` implements the same affine map (None == no normalization)."""
        if other is None:
            return False
        if (self.action_dim != other.action_dim or self.clip != other.clip
                or self.clip_dims != other.clip_dims or self.layout != other.layout):
            return False
        return bool(
            torch.allclose(self.q01, other.q01.to(self.q01), atol=atol) and
            torch.allclose(self.q99, other.q99.to(self.q99), atol=atol)
        )


def build_action_normalizer(stats: Optional[str | dict], what: str = "action stats",
                            expect_layout: str = ACTION_LAYOUT):
    """Make an `ActionNormalizer` from a JSON path or an already-loaded dict.

    Returns None for a None/empty input, which is the "no normalization" mode and
    reproduces the behaviour of this repo before q01/q99 normalization existed.

    `expect_layout` should be the configured `ActionSpace.layout`; a stats file stamped
    with anything else is refused.
    """
    if stats is None or stats == "":
        return None
    if isinstance(stats, ActionNormalizer):
        if stats.layout != expect_layout:
            raise ValueError("{}: normalizer layout '{}' but the model uses '{}'"
                             .format(what, stats.layout, expect_layout))
        return stats
    if isinstance(stats, dict):
        return ActionNormalizer.from_dict(stats, what=what, expect_layout=expect_layout)
    if isinstance(stats, str):
        return ActionNormalizer.from_json(stats, expect_layout=expect_layout)
    raise TypeError("unsupported action stats type: {}".format(type(stats)))


if __name__ == "__main__":
    # round-trip: unnormalize(normalize(x)) == x for anything inside [q01, q99]
    torch.manual_seed(0)
    # Rotation quantiles shaped like real ones: a near-identity delta puts r00/r11 close
    # to 1 and the other four close to 0, so the six channels have genuinely different
    # ranges. Using one range for all six would make the test degenerate for reasons
    # that have nothing to do with the code under test.
    q01 = [-0.05] * 3 + [0.97, -0.08, -0.06, -0.09, 0.96, -0.07] + [0.0]
    q99 = [0.05] * 3 + [1.00, 0.08, 0.06, 0.09, 1.00, 0.07] + [1.0]
    norm = ActionNormalizer(q01, q99, clip=True)
    print(norm)

    lo = torch.tensor(q01)
    hi = torch.tensor(q99)
    x = lo + (hi - lo) * torch.rand(4, 32, 10)
    y = norm.normalize(x)
    assert y.min() >= -1.0 and y.max() <= 1.0, (y.min(), y.max())
    assert torch.allclose(norm.unnormalize(y), x, atol=1e-5), (norm.unnormalize(y) - x).abs().max()

    # Saturation behaviour, the reason DEFAULT_CLIP_DIMS excludes the rotation. An
    # out-of-range model output must still clip on the metric channels and must still
    # yield a valid rotation.
    from .layers.rot_transforms import rotation_6d_to_matrix

    def orth_err(action_10):
        R = rotation_6d_to_matrix(action_10[:, 3:9])
        return (R @ R.transpose(-1, -2) - torch.eye(3)).abs().max().item()

    over = torch.full((8, 10), 7.0)  # far outside [-1, 1] on every channel
    saturated = norm.unnormalize(over)
    assert saturated[:, :3].max() <= 0.05 + 1e-6, "translation must still be clipped"
    assert saturated[:, 9].max() <= 1.0 + 1e-6, "openness must still be clipped"
    assert orth_err(saturated) < 1e-4, "saturated rotation is not a rotation"

    # ... and the same normalizer with the rotation *included* in clip_dims is exactly
    # the failure being avoided: every out-of-range 6D vector collapses onto the single
    # corner (q99_3..q99_8), so a1 and a2 are one fixed pair for all of them. Whether
    # that pair is parallel is luck, and here it is not -- but the map is no longer
    # injective, which is the actual problem: a whole half-space of model outputs
    # becomes one rotation.
    clipped_rot = ActionNormalizer(q01, q99, clip=True, clip_dims=range(10))
    a = clipped_rot.unnormalize(over)
    b = clipped_rot.unnormalize(over * 3)
    assert torch.allclose(a, b), "clipping the rotation should collapse distinct outputs"
    assert not torch.allclose(norm.unnormalize(over), norm.unnormalize(over * 3)), \
        "without rotation clipping the map must stay injective"

    # the 9-dim geometry prefix uses the same channels
    assert torch.allclose(norm.unnormalize(norm.normalize(x[..., :9])), x[..., :9], atol=1e-5)

    # a constant channel must survive rather than divide by zero
    flat = ActionNormalizer([0.0] * 10, [0.0] * 3 + [1.0] * 7, clip=False)
    z = torch.randn(2, 5, 10)
    assert torch.allclose(flat.unnormalize(flat.normalize(z)), z, atol=1e-5)
    assert torch.allclose(flat.normalize(z)[..., :3], z[..., :3]), "constant dims pass through"

    # JSON round trip
    d = norm.to_dict()
    assert norm.matches(ActionNormalizer.from_dict(d))
    print("[OK] action_norm self-test passed")
