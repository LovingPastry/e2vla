"""Migrate a joint checkpoint from normalized openness to raw gripper values.

Only affine metadata changes; model weights are copied byte-for-byte.  The required
old endpoints describe the transform used while training the old checkpoint:

    openness = (raw_gripper - old_raw_min) / (old_raw_max - old_raw_min)
    legacy_action = 2 * openness - 1

After migration, q01/q99 absorb that transform and ``action2states`` returns
``raw_gripper`` directly.
"""

import argparse
import os
import re

import torch

from models.action_norm import STATS_VERSION


OLD_LAYOUT_RE = re.compile(r"^abs_joint(\d+)_openness$")


def migrate_checkpoint(ckpt: dict, old_raw_min: float, old_raw_max: float) -> dict:
    if not old_raw_max > old_raw_min:
        raise ValueError("old_raw_max must be greater than old_raw_min")

    old_layout = ckpt.get("action_layout")
    match = OLD_LAYOUT_RE.fullmatch(str(old_layout))
    if match is None:
        raise ValueError(
            "expected an old joint layout like 'abs_joint7_openness', got {!r}"
            .format(old_layout))

    num_joints = int(match.group(1))
    action_dim = num_joints + 1
    new_layout = "abs_joint{}_raw_gripper".format(num_joints)
    span = old_raw_max - old_raw_min
    old_stats = ckpt.get("action_norm")

    if old_stats is None:
        # Identity on joint channels and the old [-1,1] gripper encoding on the last
        # channel. clip=False is required to preserve the old no-normalizer path exactly.
        q01 = [-1.0] * action_dim
        q99 = [1.0] * action_dim
        q01[-1] = old_raw_min
        q99[-1] = old_raw_max
        new_stats = {
            "version": STATS_VERSION,
            "layout": new_layout,
            "action_dim": action_dim,
            "clip": False,
            "clip_dims": list(range(action_dim)),
            "q01": q01,
            "q99": q99,
            "meta": {"migration": "legacy openness without action_norm"},
        }
    else:
        if old_stats.get("layout") != old_layout:
            raise ValueError(
                "checkpoint action_layout {!r} disagrees with action_norm layout {!r}"
                .format(old_layout, old_stats.get("layout")))
        if int(old_stats.get("action_dim", len(old_stats["q01"]))) != action_dim:
            raise ValueError("checkpoint action_norm has the wrong action_dim")

        new_stats = dict(old_stats)
        q01 = list(old_stats["q01"])
        q99 = list(old_stats["q99"])
        # raw = (legacy_action + 1) / 2 * span + old_raw_min
        q01[-1] = (float(q01[-1]) + 1.0) * 0.5 * span + old_raw_min
        q99[-1] = (float(q99[-1]) + 1.0) * 0.5 * span + old_raw_min
        new_stats["layout"] = new_layout
        new_stats["q01"] = q01
        new_stats["q99"] = q99
        meta = dict(new_stats.get("meta", {}))
        meta["migration"] = {
            "from_layout": old_layout,
            "old_raw_min": old_raw_min,
            "old_raw_max": old_raw_max,
        }
        new_stats["meta"] = meta

    migrated = dict(ckpt)
    migrated["action_layout"] = new_layout
    migrated["action_norm"] = new_stats
    return migrated


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="old checkpoint")
    parser.add_argument("--output", required=True, help="new checkpoint; must not exist")
    parser.add_argument("--old-raw-min", required=True, type=float)
    parser.add_argument("--old-raw-max", required=True, type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    if input_path == output_path:
        raise ValueError("refusing to overwrite the source checkpoint")
    if os.path.exists(output_path):
        raise FileExistsError("output already exists: {}".format(output_path))

    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)
    migrated = migrate_checkpoint(ckpt, args.old_raw_min, args.old_raw_max)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(migrated, output_path)
    print("[OK] {} -> {}".format(input_path, output_path))
    print("[OK] action layout: {}".format(migrated["action_layout"]))
    print("[OK] raw gripper q01/q99: {:.8f} / {:.8f}".format(
        migrated["action_norm"]["q01"][-1],
        migrated["action_norm"]["q99"][-1]))
    print("[NOTE] Keep legacy_gripper_scale=1.0 in the checkpoint directory config.")


if __name__ == "__main__":
    main()
