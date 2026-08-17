# E2VLA 真机关节模型训练

模型直接预测：

```text
[q0, q1, q2, q3, q4, q5, q6, gripper_raw]
```

`joint[:, -1]` 是夹爪原值。它和关节角一样，只使用当前数据计算出的 q01/q99 归一化。新训练不使用 `GRIPPER_MIN`、`GRIPPER_MAX`、`norm_openness` 或 `legacy_gripper_scale`。

数据和 GPU 位于服务器。以下命令都要在服务器的仓库根目录运行。

旧 checkpoint 的迁移方法见 [旧 checkpoint 兼容](docs/LEGACY_CHECKPOINTS.md)。

## 1. 新增配置

先在 `data_utils/dataset_real.py` 中确认：

```python
ACTION_SPACE = "joint7"
NUM_JOINTS = 7
```

数据中的 `joint` 必须有 8 列：前 7 列是弧度制关节角，最后一列是夹爪原值。

在 `configs.py` 的 `CONFIGS` 区域新增配置：

```python
CONFIGS["va_real_joint_new"] = make_real_joint_config(
    context_encoder="sa",
    action_norm_stats="./action_stats/real_joint7_new.json",
    pretrained_ckpt=None,
    legacy_gripper_scale=1.0,
)
```

对算法的修改可以参考 [算法配置讲解](docs/ALGORITHMS.md)。

检查数据：

```bash
python -m data_utils.dataset_real /data/lanzc/task0_0716_process
```

## 2. 计算 norm states

使用训练配置重新计算 action q01/q99：

```bash
python -m data_prepare.compute_action_stats \
  --config va_real_joint_new \
  -o ./action_stats/real_joint7_new.json
```

输出必须显示：

```text
layout = abs_joint7_raw_gripper
action_dim = 8
```

统计文件的最后一维必须来自 `joint[:, -1]`。更换数据、关节数或采样配置后，需要重新计算。

数据和动作空间相同时，`sa`、`transformer`、`mlp`、DDIM 和 Flow 可以共享同一份统计文件。

## 3. 训练

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config va_real_joint_new \
  -s VA_REAL_JOINT_NEW
```

临时覆盖 batch size、学习率和训练步数：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config va_real_joint_new \
  --bs 8 \
  --max_lr 5e-5 \
  --max_iterations 30000 \
  -s VA_REAL_JOINT_NEW
```

输出位置：

```text
logs/E2VLA/VA_REAL_JOINT_NEW/
checkpoints/E2VLA/VA_REAL_JOINT_NEW/
```

查看曲线：

```bash
tensorboard --logdir ./logs/E2VLA
```

## 4. 开环测评

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --ckpt ./checkpoints/E2VLA/VA_REAL_JOINT_NEW/ckpt_latest.pt \
  --data_root /data/lanzc/task0_0716_process \
  --num_samples 256 \
  --bs 8 \
  --workers 4 \
  --seed 0 \
  --ema \
  --save
```

主要查看：

```text
joint_err_rad
joint_err_deg
grip_l1
```

`grip_l1` 的单位与磁盘中的 `joint[:, -1]` 相同。结果写入 `eval_results/`。
