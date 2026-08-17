"""Migrate a joint checkpoint from the legacy misnamed channel to raw gripper values.

Only affine metadata changes; model weights are copied byte-for-byte.  The required
old endpoints describe the transform used while training the old checkpoint:

    legacy_unit = (raw_gripper - old_raw_min) / (old_raw_max - old_raw_min)
    legacy_action = 2 * legacy_unit - 1

Despite the old ``*_openness`` layout name, the real joint data use 0=open and larger
values=close. The formula never inverts that direction; only the historical name was
wrong.

After migration, q01/q99 absorb that transform and ``action2states`` returns
``raw_gripper`` directly.

Usage：
python -m data_prepare.migrate_joint_gripper_checkpoint \
  --input checkpoints/E2VA/VA_TF_FLOW/ckpt_best.pt \
  --output checkpoints/E2VA/VA_TF_FLOW_RAW/ckpt_best.pt \
  --old-raw-min 0.0 \
  --old-raw-max 1.5 \
  --config checkpoints/E2VA/VA_TF_FLOW/202608121001.json \
  --output-config checkpoints/E2VA/VA_TF_FLOW_RAW/config.json

迁移完成后，在服务器仓库根目录依次验证（GT 是磁盘原始 action，Pred 是模型最终返回值）：

# 1. 固定一个观测，逐步打印一个 action chunk；--raw 同时打印模型内部归一化值
CUDA_VISIBLE_DEVICES=0 python dump_chunk.py \
  --ckpt checkpoints/E2VA/VA_TF_FLOW_RAW/ckpt_best.pt \
  --data_root /data/lanzc/task0_0716_process \
  --index 0 --step 0 --samples 3 --raw

# 2. 扫完整 episode；生成的 CSV 中 gt=原始 actions[:, -1]，pred=最终模型输出
CUDA_VISIBLE_DEVICES=0 python gripper_curve.py \
  --ckpt checkpoints/E2VA/VA_TF_FLOW_RAW/ckpt_best.pt \
  --data_root /data/lanzc/task0_0716_process \
  --episode 0 --exec_step 0 --stride 1 --bs 8 --no_table \
  --out eval_results/gripper_raw_ep0.png

# 3. 批量开环统计；grip_l1 的单位与磁盘原始夹爪 action 相同
CUDA_VISIBLE_DEVICES=0 python test.py \
  --ckpt checkpoints/E2VA/VA_TF_FLOW_RAW/ckpt_best.pt \
  --data_root /data/lanzc/task0_0716_process \
  --num_samples 256 --bs 8 --workers 4 --seed 0 --save

同一训练目录下迁移多个 checkpoint 时，为每个 --output 使用不同文件名，但可以重复传入
同一个 --output-config。脚本会复用内容一致的 config.json 和 action stats；若内容不同则拒绝，
防止把不同实验的权重与配置混在一起。例如继续迁移 latest：

python -m data_prepare.migrate_joint_gripper_checkpoint \
  --input checkpoints/E2VA/VA_TF_FLOW/ckpt_latest.pt \
  --output checkpoints/E2VA/VA_TF_FLOW_RAW/ckpt_latest.pt \
  --old-raw-min 0.0 --old-raw-max 1.5 \
  --config checkpoints/E2VA/VA_TF_FLOW/202608121001.json \
  --output-config checkpoints/E2VA/VA_TF_FLOW_RAW/config.json
"""

import argparse
import json
import os
import re

import torch

from models.action_norm import STATS_VERSION


OLD_LAYOUT_RE = re.compile(r"^abs_joint(\d+)_openness$")


def write_or_reuse_json(path: str, payload: dict, label: str) -> str:
    """Create shared metadata once, or verify an existing copy is identical.

    Several checkpoints from one run have the same model config and action statistics.
    Requiring a fresh JSON filename for every weight file is both noisy and dangerous:
    inference selects configuration by directory, not by checkpoint basename.
    """
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fp:
                existing = json.load(fp)
        except (OSError, ValueError) as exc:
            raise FileExistsError(
                "{} exists but is not a readable JSON file: {}".format(label, path)
            ) from exc
        if existing != payload:
            raise FileExistsError(
                "{} already exists with different content: {}. Use a separate output "
                "directory/config for checkpoints from a different experiment."
                .format(label, path))
        return "reused"

    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=4)
    return "wrote"


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
    parser.add_argument("--config", help="config JSON next to the old checkpoint")
    parser.add_argument("--output-config",
                        help="write a migrated config; requires --config")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    if input_path == output_path:
        raise ValueError("refusing to overwrite the source checkpoint")
    if os.path.exists(output_path):
        raise FileExistsError("output already exists: {}".format(output_path))
    if bool(args.config) != bool(args.output_config):
        raise ValueError("--config and --output-config must be provided together")
    config_input = config_output = stats_path = config = None
    if args.config:
        config_input = os.path.abspath(args.config)
        config_output = os.path.abspath(args.output_config)
        if config_input == config_output:
            raise ValueError("refusing to overwrite the source config")
        config_dir = os.path.dirname(config_output)
        stats_path = os.path.join(config_dir, "raw_gripper_action_norm.json")
        with open(config_input, "r", encoding="utf-8") as fp:
            config = json.load(fp)

    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)
    migrated = migrate_checkpoint(ckpt, args.old_raw_min, args.old_raw_max)

    # Validate/reuse shared metadata before creating the checkpoint. If a caller points a
    # checkpoint from another experiment at this directory, fail without leaving behind
    # a weight file that inference could later pair with the wrong config.
    if args.config:
        config["legacy_gripper_scale"] = 1.0

        os.makedirs(config_dir, exist_ok=True)
        config["action_norm_stats"] = stats_path
        stats_status = write_or_reuse_json(
            stats_path, migrated["action_norm"], "output action stats")
        config_status = write_or_reuse_json(
            config_output, config, "output config")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(migrated, output_path)

    print("[OK] {} -> {}".format(input_path, output_path))
    print("[OK] action layout: {}".format(migrated["action_layout"]))
    print("[OK] raw gripper q01/q99: {:.8f} / {:.8f}".format(
        migrated["action_norm"]["q01"][-1],
        migrated["action_norm"]["q99"][-1]))
    if args.config:
        print("[OK] migrated config ({}): {}".format(config_status, config_output))
        print("[OK] migrated action stats ({}): {}".format(
            stats_status, stats_path))
    else:
        print("[NOTE] Keep legacy_gripper_scale=1.0 in the checkpoint directory config.")


if __name__ == "__main__":
    main()
