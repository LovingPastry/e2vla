"""Audit every affine transform in the legacy real-joint gripper pipeline.

The script is read-only. It scans memmap episodes, reads an old checkpoint and its
neighboring config JSON files, then reconstructs this exact chain:

    raw data -> legacy unit channel -> [-1, 1] -> q01/q99 -> inverse q01/q99
             -> legacy unit channel -> legacy_gripper_scale -> robot command

Run from the repository root on the data/compute server. The defaults encode the known
old training calibration (GRIPPER_MIN=0.0, GRIPPER_MAX=1.5) and the observed robot
command interval [0.0, 0.8].

Usage：
python -m data_prepare.audit_gripper_pipeline \
  --data-root /data/lanzc/task0_0716_process \
  --ckpt /path/to/old_checkpoint.pt \
  --old-gripper-min 0.0 \
  --old-gripper-max 1.5 \
  --robot-command-min 0.0 \
  --robot-command-max 0.8 \
  --output ./gripper_pipeline_audit.json

"""

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def describe(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"count": int(len(values)), "finite": 0}
    q = np.quantile(finite, [0.0, 0.01, 0.5, 0.99, 1.0])
    return {
        "count": int(len(values)),
        "finite": int(len(finite)),
        "min": float(q[0]),
        "q01": float(q[1]),
        "median": float(q[2]),
        "q99": float(q[3]),
        "max": float(q[4]),
    }


def error_summary(actual: np.ndarray, expected: np.ndarray) -> dict:
    delta = np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(delta))),
        "q99_abs": float(np.quantile(np.abs(delta), 0.99)),
        "max_abs": float(np.max(np.abs(delta))),
    }


def load_array(traj_dir: str, name: str, metadata: dict) -> np.ndarray:
    attr = metadata[name]
    return np.memmap(
        os.path.join(traj_dir, name + ".bin"), mode="r",
        dtype=attr["dtype"], shape=tuple(attr["shape"]))


def affine_fit(x: np.ndarray, y: np.ndarray) -> dict:
    design = np.stack([x, np.ones_like(x)], axis=1)
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = slope * x + intercept
    result = {
        "formula": "y ~= slope * x + intercept",
        "slope": float(slope),
        "intercept": float(intercept),
        "rmse": float(np.sqrt(np.mean((y - fitted) ** 2))),
        "max_abs_residual": float(np.max(np.abs(y - fitted))),
    }
    if abs(slope) > 1e-12:
        result["x_at_y_0"] = float(-intercept / slope)
        result["x_at_y_1"] = float((1.0 - intercept) / slope)
    return result


def scan_data(data_root: str, episode_limit: int,
              old_min: float, old_max: float) -> Tuple[dict, dict]:
    metadata_paths = sorted(glob.glob(
        os.path.join(os.path.abspath(data_root), "**", "metadata.json"),
        recursive=True))
    if episode_limit > 0:
        metadata_paths = metadata_paths[:episode_limit]
    if not metadata_paths:
        raise FileNotFoundError("no metadata.json under {}".format(data_root))

    joint_parts: List[np.ndarray] = []
    action_parts: List[np.ndarray] = []
    paired_joint_actions: List[Tuple[np.ndarray, np.ndarray]] = []
    norm_parts: List[np.ndarray] = []
    paired_joint_norm: List[Tuple[np.ndarray, np.ndarray]] = []
    legacy_unit_parts: List[np.ndarray] = []
    legacy_joint_parts: List[np.ndarray] = []
    source_counts = {"norm_openness": 0, "joint_affine": 0}
    skipped = []
    old_span = old_max - old_min

    for metadata_path in metadata_paths:
        traj_dir = os.path.dirname(metadata_path)
        with open(metadata_path, "r", encoding="utf-8") as fp:
            metadata = json.load(fp)
        if "joint" not in metadata:
            skipped.append({"episode": traj_dir, "reason": "missing joint"})
            continue

        joint = np.asarray(load_array(traj_dir, "joint", metadata), dtype=np.float64)
        if joint.ndim != 2 or joint.shape[1] < 2:
            skipped.append({"episode": traj_dir,
                            "reason": "invalid joint shape {}".format(joint.shape)})
            continue
        raw_joint = joint[:, -1]
        joint_parts.append(raw_joint)

        action_name = next((name for name in ("actions", "action")
                            if name in metadata), None)
        if action_name is not None:
            actions = np.asarray(load_array(traj_dir, action_name, metadata),
                                 dtype=np.float64)
            if actions.ndim == 2 and len(actions) == len(raw_joint):
                raw_action = actions[:, -1]
                action_parts.append(raw_action)
                paired_joint_actions.append((raw_joint, raw_action))

        # This reproduces the OLD RealBinDataset.get_openness branch order exactly:
        # norm_openness won whenever the field existed; otherwise joint[:, -1] was
        # mapped by GRIPPER_MIN/MAX and clipped.
        if "norm_openness" in metadata:
            legacy_unit = np.asarray(
                load_array(traj_dir, "norm_openness", metadata),
                dtype=np.float64).reshape(-1)
            if len(legacy_unit) != len(raw_joint):
                skipped.append({"episode": traj_dir,
                                "reason": "norm_openness length mismatch"})
                continue
            norm_parts.append(legacy_unit)
            paired_joint_norm.append((raw_joint, legacy_unit))
            source_counts["norm_openness"] += 1
        else:
            legacy_unit = np.clip((raw_joint - old_min) / old_span, 0.0, 1.0)
            source_counts["joint_affine"] += 1
        legacy_unit_parts.append(legacy_unit)
        legacy_joint_parts.append(raw_joint)

    if not legacy_unit_parts:
        raise RuntimeError("no usable episodes")

    joint = np.concatenate(joint_parts)
    legacy_unit = np.concatenate(legacy_unit_parts)
    legacy_joint = np.concatenate(legacy_joint_parts)
    public = {
        "root": os.path.abspath(data_root),
        "episodes_scanned": len(metadata_paths),
        "episodes_used": int(sum(source_counts.values())),
        "legacy_source_episode_counts": source_counts,
        "joint_last": describe(joint),
        "legacy_unit_before_model": describe(legacy_unit),
        "skipped": skipped,
    }

    arrays = {"joint": joint, "legacy_joint": legacy_joint,
              "legacy_unit": legacy_unit}
    if action_parts:
        action = np.concatenate(action_parts)
        public["actions_last"] = describe(action)
        arrays["actions"] = action
    if paired_joint_actions:
        paired_joint = np.concatenate([x for x, _ in paired_joint_actions])
        paired_action = np.concatenate([y for _, y in paired_joint_actions])
        public["actions_last_vs_joint_last"] = {
            "error_if_treated_as_same_value": error_summary(paired_action, paired_joint),
            "affine_fit_action_from_joint": affine_fit(paired_joint, paired_action),
            "correlation": float(np.corrcoef(paired_joint, paired_action)[0, 1]),
        }
    if norm_parts:
        norm = np.concatenate(norm_parts)
        public["norm_openness"] = describe(norm)
    if paired_joint_norm:
        paired_joint = np.concatenate([x for x, _ in paired_joint_norm])
        paired_norm = np.concatenate([y for _, y in paired_joint_norm])
        public["norm_openness_from_joint_fit"] = affine_fit(paired_joint, paired_norm)
    return public, arrays


def neighboring_configs(ckpt_path: str) -> List[dict]:
    configs = []
    for path in sorted(glob.glob(os.path.join(os.path.dirname(ckpt_path), "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except (OSError, ValueError):
            continue
        if "legacy_gripper_scale" in data or "action_space" in data:
            configs.append({
                "path": os.path.abspath(path),
                "legacy_gripper_scale": data.get("legacy_gripper_scale"),
                "action_space": data.get("action_space"),
                "action_norm_stats": data.get("action_norm_stats"),
            })
    return configs


def resolve_legacy_scale(configs: List[dict], override: Optional[float]) -> Tuple[float, str]:
    if override is not None:
        return float(override), "--legacy-scale"
    values = sorted(set(float(c["legacy_gripper_scale"]) for c in configs
                        if isinstance(c.get("legacy_gripper_scale"), (int, float))))
    if not values:
        return 1.0, "default (missing from neighboring configs)"
    if len(values) > 1:
        raise ValueError(
            "neighboring configs contain multiple legacy_gripper_scale values {}; "
            "pass --legacy-scale explicitly".format(values))
    return values[0], "neighboring config"


def checkpoint_stats(ckpt: dict) -> dict:
    stats = ckpt.get("action_norm")
    result = {
        "action_layout": ckpt.get("action_layout", "<missing>"),
        "objective": ckpt.get("objective", "<missing>"),
        "has_action_norm": stats is not None,
    }
    if stats is not None:
        result["action_norm"] = {
            "layout": stats.get("layout"),
            "clip": bool(stats.get("clip", True)),
            "clip_dims": stats.get("clip_dims"),
            "gripper_q01": float(stats["q01"][-1]),
            "gripper_q99": float(stats["q99"][-1]),
            "meta": stats.get("meta", {}),
        }
    return result


def q_roundtrip(values: np.ndarray, stats: Optional[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (normalized model value, inverse-normalized legacy action)."""
    if stats is None:
        return values.copy(), values.copy()
    q01 = float(stats["q01"][-1])
    q99 = float(stats["q99"][-1])
    span = q99 - q01
    if span < 1e-6:
        return values.copy(), values.copy()
    normalized = 2.0 * (values - q01) / span - 1.0
    clip_dims = stats.get("clip_dims")
    last_dim = int(stats.get("action_dim", len(stats["q01"]))) - 1
    clips_gripper = bool(stats.get("clip", True)) and (
        clip_dims is None or last_dim in clip_dims)
    if clips_gripper:
        normalized = np.clip(normalized, -1.0, 1.0)
    restored = (normalized + 1.0) * 0.5 * span + q01
    return normalized, restored


def build_pipeline_report(arrays: dict, ckpt: dict, old_min: float, old_max: float,
                          legacy_scale: float, scale_source: str,
                          robot_min: float, robot_max: float) -> dict:
    legacy_unit = arrays["legacy_unit"]
    raw_joint = arrays["legacy_joint"]
    legacy_action = 2.0 * legacy_unit - 1.0
    model_value, restored_action = q_roundtrip(legacy_action, ckpt.get("action_norm"))
    restored_unit = (restored_action + 1.0) * 0.5
    ideal_raw_inverse = legacy_unit * (old_max - old_min) + old_min
    correct_raw_inverse = restored_unit * (old_max - old_min) + old_min
    legacy_scaled = np.clip(restored_unit * legacy_scale, 0.0, 1.0)
    robot_command = robot_min + legacy_scaled * (robot_max - robot_min)
    legacy_clip_fraction = float(np.mean(
        (restored_unit * legacy_scale < 0.0)
        | (restored_unit * legacy_scale > 1.0)))

    old_stats = ckpt.get("action_norm")
    if old_stats is None:
        # The old model denoised joint angles directly and the gripper's hard-coded
        # 2*unit-1 value directly. The migration normalizer must therefore be identity
        # on joint channels and [old_min, old_max] on the raw gripper channel.
        migrated_stats = {
            "raw_gripper_q01": old_min,
            "raw_gripper_q99": old_max,
            "note": ("compatibility stats for this trained checkpoint; do not replace "
                     "with empirical data q01/q99 without retraining"),
        }
    else:
        migrated_stats = {
            "raw_gripper_q01": ((float(old_stats["q01"][-1]) + 1.0) * 0.5
                                  * (old_max - old_min) + old_min),
            "raw_gripper_q99": ((float(old_stats["q99"][-1]) + 1.0) * 0.5
                                  * (old_max - old_min) + old_min),
        }

    result = {
        "known_old_calibration": {"GRIPPER_MIN": old_min, "GRIPPER_MAX": old_max},
        "legacy_gripper_scale": legacy_scale,
        "legacy_gripper_scale_source": scale_source,
        "assumed_robot_command_range": [robot_min, robot_max],
        "formulas": {
            "old_raw_to_unit": "unit=(raw-{:.9g})/{:.9g}".format(
                old_min, old_max - old_min),
            "old_unit_to_action": "legacy_action=2*unit-1",
            "q_normalize": "model=2*(legacy_action-q01)/(q99-q01)-1",
            "correct_old_inverse": "raw=(legacy_action+1)/2*{:.9g}+{:.9g}".format(
                old_max - old_min, old_min),
            "legacy_postprocess": "clip(unit*{:.9g},0,1)".format(legacy_scale),
            "assumed_robot_map": "robot={:.9g}+legacy*{:.9g}".format(
                robot_min, robot_max - robot_min),
        },
        "legacy_scale_clip_fraction": legacy_clip_fraction,
        "stages": {
            "01_legacy_unit": describe(legacy_unit),
            "02_legacy_action_2x_minus_1": describe(legacy_action),
            "03_model_normalized_value": describe(model_value),
            "04_after_q01_q99_inverse": describe(restored_action),
            "05_decoded_legacy_unit": describe(restored_unit),
            "06_correct_inverse_to_raw_action": describe(correct_raw_inverse),
            "07_after_legacy_scale_and_clip": describe(legacy_scaled),
            "08_if_mapped_to_robot_command_range": describe(robot_command),
        },
        "roundtrip_checks": {
            "q01_q99_action_roundtrip": error_summary(restored_action, legacy_action),
            "old_unit_affine_inverse_vs_joint_last_before_q_clipping": error_summary(
                ideal_raw_inverse, raw_joint),
            "q_clipping_effect_in_raw_units": error_summary(
                correct_raw_inverse, ideal_raw_inverse),
            "correct_raw_inverse_vs_joint_last": error_summary(
                correct_raw_inverse, raw_joint),
        },
        "new_raw_pipeline_parameters": {
            "joint_last_direct_q01": describe(arrays["joint"])["q01"],
            "joint_last_direct_q99": describe(arrays["joint"])["q99"],
            "migrated_checkpoint_q01_q99": migrated_stats,
            "correct_legacy_decode_to_raw": {
                "scale": old_max - old_min,
                "offset": old_min,
                "formula": "raw=decoded_unit*scale+offset",
            },
            "legacy_gripper_scale_after_checkpoint_migration": 1.0,
            "old_code_hotfix_without_checkpoint_migration": {
                "legacy_gripper_scale": old_max - old_min,
                "valid_only_because_old_gripper_min_is_zero": old_min == 0.0,
                "robot_side_extra_0_to_1_mapping": "remove",
            },
            "extra_openness_or_robot_range_mapping": "none",
        },
    }
    if "actions" in arrays:
        action_desc = describe(arrays["actions"])
        result["new_raw_pipeline_parameters"]["actions_last_direct_q01"] = action_desc["q01"]
        result["new_raw_pipeline_parameters"]["actions_last_direct_q99"] = action_desc["q99"]

    verdicts = []
    if legacy_scale != 1.0:
        verdicts.append(
            "legacy_gripper_scale changes the post-inverse value; it is not the inverse "
            "of the old raw->unit affine unless paired with the exact old endpoints")
    if legacy_clip_fraction > 0:
        verdicts.append("legacy scale clips {:.2%} of samples and is not invertible"
                        .format(legacy_clip_fraction))
    mapping_error = result["roundtrip_checks"][
        "old_unit_affine_inverse_vs_joint_last_before_q_clipping"]["mae"]
    if mapping_error > 1e-4:
        verdicts.append(
            "old unit values do not invert back to joint[:, -1] under 0.0/1.5; "
            "norm_openness or a different supervision source changed the mapping")
    else:
        verdicts.append(
            "0.0/1.5 inverse recovers joint[:, -1] apart from q01/q99 tail clipping")
    result["verdicts"] = verdicts
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--episodes", type=int, default=0,
                        help="0 scans every episode")
    parser.add_argument("--old-gripper-min", type=float, default=0.0)
    parser.add_argument("--old-gripper-max", type=float, default=1.5)
    parser.add_argument("--legacy-scale", type=float,
                        help="override config legacy_gripper_scale")
    parser.add_argument("--robot-command-min", type=float, default=0.0)
    parser.add_argument("--robot-command-max", type=float, default=0.8)
    parser.add_argument("--output", help="optional JSON output path")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.old_gripper_max > args.old_gripper_min:
        raise ValueError("old-gripper-max must be greater than old-gripper-min")
    if not args.robot_command_max > args.robot_command_min:
        raise ValueError("robot-command-max must be greater than robot-command-min")

    ckpt_path = os.path.abspath(args.ckpt)
    configs = neighboring_configs(ckpt_path)
    legacy_scale, scale_source = resolve_legacy_scale(configs, args.legacy_scale)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    data_report, arrays = scan_data(
        args.data_root, args.episodes,
        args.old_gripper_min, args.old_gripper_max)
    report = {
        "data": data_report,
        "checkpoint": checkpoint_stats(ckpt),
        "neighboring_configs": configs,
        "pipeline": build_pipeline_report(
            arrays, ckpt, args.old_gripper_min, args.old_gripper_max,
            legacy_scale, scale_source,
            args.robot_command_min, args.robot_command_max),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output_path = os.path.abspath(args.output)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fp:
            fp.write(rendered + "\n")
        print("[OK] wrote {}".format(output_path))


if __name__ == "__main__":
    main()
