"""Read-only inspection of raw gripper data and legacy checkpoint semantics.

This script intentionally imports no project modules, needs no CUDA, and never writes
data or checkpoints. Run it from the repository root and send back its complete output.
"""

import argparse
import glob
import json
import os

import numpy as np
import torch


def describe(values: np.ndarray):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"count": int(len(values)), "finite": 0}
    quantiles = np.quantile(finite, [0.0, 0.01, 0.5, 0.99, 1.0])
    return {
        "count": int(len(values)),
        "finite": int(len(finite)),
        "min": float(quantiles[0]),
        "q01": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q99": float(quantiles[3]),
        "max": float(quantiles[4]),
    }


def load_array(traj_dir: str, name: str, metadata: dict):
    attr = metadata[name]
    return np.memmap(
        os.path.join(traj_dir, name + ".bin"), mode="r",
        dtype=attr["dtype"], shape=tuple(attr["shape"]))


def inspect_data(data_root: str, episode_limit: int):
    metadata_paths = sorted(glob.glob(
        os.path.join(os.path.abspath(data_root), "**", "metadata.json"),
        recursive=True))
    if episode_limit > 0:
        metadata_paths = metadata_paths[:episode_limit]
    if not metadata_paths:
        raise FileNotFoundError("no metadata.json under {}".format(data_root))

    raw_parts = []
    norm_parts = []
    paired_raw = []
    paired_norm = []
    schemas = {}
    episodes_with_norm = 0

    for metadata_path in metadata_paths:
        traj_dir = os.path.dirname(metadata_path)
        with open(metadata_path, "r", encoding="utf-8") as fp:
            metadata = json.load(fp)
        schema = tuple(sorted(
            # Episode length (axis 0) naturally varies; omit it so the report stays
            # compact and only real schema differences create another variant.
            (name, ("T",) + tuple(attr["shape"][1:]), str(attr["dtype"]))
            for name, attr in metadata.items()))
        schemas[str(schema)] = schemas.get(str(schema), 0) + 1
        if "joint" not in metadata:
            raise KeyError("{} has no 'joint' entry".format(metadata_path))

        joint = load_array(traj_dir, "joint", metadata)
        if joint.ndim != 2 or joint.shape[1] < 2:
            raise ValueError("{}: joint shape is {}".format(traj_dir, joint.shape))
        raw = np.asarray(joint[:, -1], dtype=np.float64)
        raw_parts.append(raw)

        if "norm_openness" in metadata:
            episodes_with_norm += 1
            norm = np.asarray(
                load_array(traj_dir, "norm_openness", metadata),
                dtype=np.float64).reshape(-1)
            norm_parts.append(norm)
            if len(norm) == len(raw):
                paired_raw.append(raw)
                paired_norm.append(norm)

    raw = np.concatenate(raw_parts)
    result = {
        "root": os.path.abspath(data_root),
        "episodes_scanned": len(metadata_paths),
        "episodes_with_norm_openness": episodes_with_norm,
        "joint_last_raw": describe(raw),
        "schema_variants": schemas,
    }

    if norm_parts:
        norm = np.concatenate(norm_parts)
        result["norm_openness"] = describe(norm)
    if paired_raw:
        x = np.concatenate(paired_raw)
        y = np.concatenate(paired_norm)
        design = np.stack([x, np.ones_like(x)], axis=1)
        slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
        fitted = slope * x + intercept
        residual = y - fitted
        fit = {
            "formula": "norm_openness ~= slope * joint_last + intercept",
            "slope": float(slope),
            "intercept": float(intercept),
            "max_abs_residual": float(np.max(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
        }
        if abs(slope) > 1e-12:
            fit["inferred_raw_at_openness_0"] = float(-intercept / slope)
            fit["inferred_raw_at_openness_1"] = float((1.0 - intercept) / slope)
        result["raw_to_norm_fit"] = fit
    return result


def inspect_checkpoint(path: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    stats = checkpoint.get("action_norm")
    result = {
        "path": os.path.abspath(path),
        "action_layout": checkpoint.get("action_layout", "<missing>"),
        "objective": checkpoint.get("objective", "<missing>"),
        "has_action_norm": stats is not None,
    }
    if stats is not None:
        result["action_norm"] = {
            "version": stats.get("version"),
            "layout": stats.get("layout"),
            "action_dim": stats.get("action_dim", len(stats.get("q01", []))),
            "clip": stats.get("clip"),
            "clip_dims": stats.get("clip_dims"),
            "gripper_q01": stats["q01"][-1],
            "gripper_q99": stats["q99"][-1],
            "meta": stats.get("meta", {}),
        }

    config_files = sorted(glob.glob(os.path.join(os.path.dirname(path), "*.json")))
    configs = []
    for config_path in config_files:
        try:
            with open(config_path, "r", encoding="utf-8") as fp:
                config = json.load(fp)
            configs.append({
                "file": os.path.basename(config_path),
                "action_space": config.get("action_space"),
                "action_norm_stats": config.get("action_norm_stats"),
                "legacy_gripper_scale": config.get("legacy_gripper_scale", "<missing>"),
                "dataset_classes": config.get("dataset_classes"),
            })
        except (OSError, ValueError) as exc:
            configs.append({"file": os.path.basename(config_path), "error": str(exc)})
    result["neighbor_configs"] = configs
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ckpt", required=True, nargs="+",
                        help="one or more old checkpoints")
    parser.add_argument("--episodes", type=int, default=0,
                        help="number of episodes to scan; 0 means all")
    return parser.parse_args()


def main():
    args = parse_args()
    report = {
        "data": inspect_data(args.data_root, args.episodes),
        "checkpoints": [inspect_checkpoint(path) for path in args.ckpt],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
