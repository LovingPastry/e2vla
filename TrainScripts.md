# 在关节空间中训练，默认配置为训练 ContexEncoder + DiT ，动作专家为 Diffusion Policy

加载官方开源的预训练模型

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config finetune_real_joint --pretrained_ckpt checkpoints/E2VLA/pretrain/0927_e2vla_base_pretrain_extra/ckpt_0600000.pt --no-pretrained-strict --pretrained-ignore-action-layout -s JointStates
```

# 加上ResNet18作为可训练的模块

## 关节空间 + Flow Matching + ResNet18 旁路（全参数训练）

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config finetune_real_joint_conv_flow \
  --pretrained_ckpt ./checkpoints/E2VLA/pretrain/0927_e2vla_base_pretrain/ckpt_0600000.pt \
  -s JointConvFlow
```

三个跨空间/跨目标的标志已经写进预设，命令行不用再传：

| | |
| --- | --- |
| `action_space` | `joint7`（数据集 `RealBinDataset`） |
| `objective` | `flow`，推理 10 步 Euler |
| `conv_tower` | `resnet18`，`conv_tower_lr_scale=2.0` |
| `lora_rank` / `vlm_lora_rank` | `0` / `0` —— ContextEncoder、DiT、ResNet18 全部参数参与训练 |
| `pretrained_strict` | `False`（跨 action layout 必须） |
| `pretrained_ignore_action_layout` | `True` |
| `pretrained_ignore_objective` | `True` |
| schedule | `5e-5` / warmup 1e3 / 30e3 步（热启动量级） |

从 ee_cam + DDIM 的预训练 ckpt 迁移过来的实测结果：**101.726M / 104.692M = 97.2%**。剩下的：

- shape mismatch 4 个（0.018M）：`hist_enc.0` / `traj_enc.0` / `act_head.3` 绑在动作编码上，这边 8 维那边 10 维
- unexpected 6 个：`dp_head.abs_pos_enc`，关节空间根本不构建（从关节角还原末端位置要正运动学）
- missing 74 个（2.951M）：ResNet18 旁路，本次新增

启动后核对一下打印的逐张量报告，mismatch 应该正好是上面那 4 个。

> **注意**：`pretrained_strict=False` 会把旁路的 missing key 和 layout 失配一起放行 —— `allow_missing_prefixes` 那条窄路径要求报告除指定前缀外是干净的，而跨 action layout 不干净。这条路上避不开。

> **注意**：关节空间的 checkpoint **只能训练，不能评测**。`infer_utils/planner.py` 无条件按 17 维 SE(3) 解码，`remote_service` 能正常构建模型，然后在 `reshape(..., 4, 4)` 处崩掉。

## 末端位姿空间的对应版本（对照组）

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config finetune_real_conv_flow \
  --pretrained_ckpt ./checkpoints/E2VLA/pretrain/0927_e2vla_base_pretrain/ckpt_0600000.pt \
  --pretrained_ignore_objective True \
  --max_lr 5e-5 --num_warmup 1000 --ema_start 1000 --max_iterations 300000 \
  -s REAL_CONV_FLOW
```

数据集是 `RealRobot`，不是 `RealBinDataset`。`finetune_real_conv_flow` 预设本身是按**从头训**配的（`1e-4` / warmup 2e3 / 60e3），所以热启动时要带上面那串 override。
