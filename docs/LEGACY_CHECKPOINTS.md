# 旧 checkpoint 兼容

本文只处理旧的 `abs_jointN_openness` checkpoint。全新训练不需要这些步骤。

旧模型曾使用固定端点把夹爪原值映射到 `[-1, 1]`。迁移工具不修改模型权重，只把旧变换折叠进 checkpoint 的 q01/q99，并把对外语义改为 `raw_gripper`。

所有命令都要在服务器的仓库根目录运行。

## 1. 审计旧链路

```bash
python -m data_prepare.audit_gripper_pipeline \
  --data-root /data/lanzc/task0_0716_process \
  --ckpt /path/to/old_checkpoint.pt \
  --old-gripper-min 0.0 \
  --old-gripper-max 1.5 \
  --robot-command-min 0.0 \
  --robot-command-max 0.8 \
  --output ./gripper_pipeline_audit.json
```

`old-gripper-min/max` 必须是旧 checkpoint 训练时使用的端点，不是新数据的 min/max。

## 2. 迁移 checkpoint

```bash
python -m data_prepare.migrate_joint_gripper_checkpoint \
  --input /path/to/old_checkpoint.pt \
  --output /path/to/migrated/ckpt_best.pt \
  --old-raw-min 0.0 \
  --old-raw-max 1.5 \
  --config /path/to/old_config.json \
  --output-config /path/to/migrated/config.json
```

迁移结果应满足：

```text
action_layout = abs_jointN_raw_gripper
legacy_gripper_scale = 1.0
```

迁移生成的 q01/q99 是旧权重的兼容参数。不要用新数据的经验 q01/q99 替换它们，除非重新训练模型。

## 3. 验证迁移结果

```bash
CUDA_VISIBLE_DEVICES=0 python dump_chunk.py \
  --ckpt /path/to/migrated/ckpt_best.pt \
  --data_root /data/lanzc/task0_0716_process \
  --index 0 \
  --step 0 \
  --samples 3 \
  --raw
```

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --ckpt /path/to/migrated/ckpt_best.pt \
  --data_root /data/lanzc/task0_0716_process \
  --num_samples 256 \
  --bs 8 \
  --workers 4 \
  --seed 0 \
  --save
```

迁移后的预测最后一维应与磁盘 `joint[:, -1]` 使用相同单位和方向。
