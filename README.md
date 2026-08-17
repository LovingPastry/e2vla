# e2vla

## 分支说明
- main：主分支，原仓库复现版，并把Diffusion Policy改成Flow Matching
- va-no-language：删去文本部分的分支
- dev：仅在服务器上有的、适配环境的分支

Diffusion Policy、Flow Matching、LoRA、ResNet18 旁路以及无语言编码器的区别和配置方法，见 [模型算法与训练配置](ALGORITHMS.md)。

# 依赖


# 用自己的数据训练

本项目当前面向单机器人、单任务的真机数据，模型直接预测绝对关节角和夹爪原始命令值，不使用末端位姿作为动作。

## 1. 准备数据

数据使用 memmap/bin 格式：每条轨迹一个目录，目录中的 `metadata.json` 描述同名 `.bin` 文件的 `dtype` 和 `shape`。数据读取实现见 `data_utils/dataset_real.py` 中的 `RealBinDataset`。

每条轨迹至少需要以下字段：

| 字段 | 形状 | 说明 |
| --- | --- | --- |
| `rgbs` | `(T, ncam, 3, H, W)` | 相机图像 |
| `joint` | `(T, nq+1)` | 前 `nq` 列为关节角（弧度），最后一列是要预测和返回的夹爪原值 |
| `ee_poses` | `(T, 4, 4)` | 当前数据契约仍需提供，仅作为观测条件，不作为监督目标 |

`norm_openness` 不再使用。数据集原样读取 `joint[:, -1]`，夹爪通道和关节角一样只经过 action statistics 里的 q01/q99 归一化。当前数据语义为 `0=夹爪张开`，数值增大表示闭合，约 `0.4=夹爪闭合`；它不是宽度，也不是 openness。模型反归一化后返回的最后一维，就是与数据文件同单位、同语义的夹爪真值。部署端不再做反向、openness 反变换或二值化。

在 `RealBinDataset` 中按实际数据修改：

- `inst()` 的 `data_root`：数据根目录。
- `CAMERA_AXIS`：`rgbs` 中的相机顺序。
- `IS_BGR`：图像是否为 BGR。
- `PROMPT_TEXT`：单任务的固定文本描述。
- `NUM_JOINTS`：机器人关节数；默认是 7。

保持 `ACTION_SPACE = "joint7"`，并确保它与 `configs.py` 中 `finetune_real_joint` 的 `action_space` 一致。当前实现按单臂读取数据，关节角必须使用弧度。

## 2. 校验数据

数据和算力仅在服务器上可用。训练前先在服务器仓库根目录运行：

```bash
python -m data_utils.dataset_real
```

该命令会检查各字段形状、图像范围、夹爪范围以及关节角量纲。

对旧 checkpoint 的完整夹爪链路做只读审计（旧训练端点默认为
`GRIPPER_MIN=0.0` / `GRIPPER_MAX=1.5`，执行器观测范围默认为 `0.0~0.8`）：

```bash
python -m data_prepare.audit_gripper_pipeline \
  --data-root /data/lanzc/task0_0716_process \
  --ckpt /path/to/old_checkpoint.pt \
  --output ./gripper_pipeline_audit.json
```

它会输出 `joint[:, -1]` / `actions[:, -1]` / `norm_openness` 的关系、checkpoint
夹爪 q01/q99、每层变换的范围和往返误差、`legacy_gripper_scale` 的裁剪比例，
以及新 raw-gripper 链路应使用的 q01/q99。

## 3. 新增训练配置

默认配置是 `configs.py` 中的 `finetune_real_joint`，由 `make_real_joint_config()` 创建。临时调整 batch size、学习率或训练步数时，直接使用命令行参数覆盖，不需要修改文件：

```bash
python train.py --config finetune_real_joint \
  --bs 8 --max_lr 5e-5 --max_iterations 30000 \
  -s EXP_NAME
```

新增配置的时候，不用复制完整的 `TrainConfig`，应从关节空间默认配置派生：

```python
CONFIGS["finetune_real_joint_small"] = make_real_joint_config(
    bs=8,
    max_lr=5e-5,
    max_iterations=int(30e3),
    action_norm_stats="./action_stats/real_joint7_small_raw_gripper.json",
)
```

新增后可通过 `--config finetune_real_joint_small` 使用。数据路径、相机顺序、关节数和夹爪量程属于数据集定义，仍应在 `RealBinDataset` 中修改，不要放进训练配置。

## 4. 计算动作归一化统计量

```bash
python -m data_prepare.compute_action_stats \
  --config finetune_real_joint \
  -o ./action_stats/real_joint7_raw_gripper.json
```

统计时的 `--config` 必须与训练时一致；如果新增了配置，就把命令中的名称换成新名称。统计文件与数据集和动作空间绑定，更换数据或关节数后需要重新计算。

旧 `abs_joint7_openness` checkpoint 不需要重训，可以只迁移夹爪通道的仿射统计和
layout 标记，模型权重不变：

```bash
python -m data_prepare.migrate_joint_gripper_checkpoint \
  --input OLD.pt --output OLD_raw_gripper.pt \
  --old-raw-min 0.0 --old-raw-max OLD_TRAINING_MAX \
  --config OLD_CONFIG.json --output-config MIGRATED_DIR/config.json
```

`OLD_TRAINING_MAX` 必须是该 checkpoint 训练时用来缩放夹爪通道的旧量程端点，不是现在数据的
max。给出 config 参数时，工具会同时写出匹配的 raw-gripper statistics，并将
`legacy_gripper_scale` 固定为 `1.0`。迁移工具拒绝覆盖任何源文件或已有输出。

## 5. 训练

从仓库根目录启动训练：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config finetune_real_joint \
  -s MY_ROBOT_JOINT_EXP
```

训练日志和 checkpoint 分别写入 `logs/E2VLA/` 和 `checkpoints/E2VLA/`。查看训练曲线：

```bash
tensorboard --logdir ./logs/E2VLA
```

## 6. 当前限制

`finetune_real_joint` 可以完成关节空间训练，但 `infer_utils/planner.py` 的部署解码仍按 17 维末端位姿实现。因此，关节空间 checkpoint 目前不能直接用于 `remote_service`，部署前需要为 planner 增加关节动作的解码与轨迹融合逻辑。
