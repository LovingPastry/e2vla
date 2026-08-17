# 模型算法与训练配置

本文说明 `configs.py` 中与模型算法有关的配置。数据路径、相机顺序、关节数和夹爪量程不属于算法配置，应在 `data_utils/dataset_real.py` 的 `RealBinDataset` 中修改。

一个训练预设主要由三个互相独立的维度组成：

| 维度 | 配置字段 | 可选值 |
| --- | --- | --- |
| 动作生成算法 | `objective` | `ddim`、`flow` |
| 训练方式 | `lora_rank`、`conv_tower` | 全量微调、LoRA、ResNet18 旁路或组合 |
| 上下文编码器 | `context_encoder` | `vl`、`sa`、`transformer`、`mlp` |

动作空间由 `action_space` 单独控制。本项目的真机配置使用 `joint7`，即模型预测 7 个绝对关节角和 1 个夹爪开合度。

## 1. 动作生成算法

Diffusion Policy 和 Flow Matching 共用同一个动作网络。两者的区别是训练目标和推理采样过程，而不是网络主体结构。

| `objective` | 训练目标 | 推理方式 | 默认推理步数 |
| --- | --- | --- | --- |
| `ddim` | 预测加入动作序列的噪声 | DDIM 反向去噪 | 20 |
| `flow` | 预测从噪声到动作的速度场 | Euler 积分 | 10 |

相关字段：

- `objective`：选择 `ddim` 或 `flow`。
- `diffusion_timesteps`：DDIM 的训练时间步数，同时定义时间嵌入的数值范围。
- `inference_timesteps`：推理时动作头的前向次数；设为 `0` 时使用上表默认值。
- `flow_time_sampling`：Flow Matching 的时间采样，可选 `uniform`、`logitnormal` 或 `beta`。
- `flow_time_alpha`：仅在 `flow_time_sampling="beta"` 时使用。

DDIM 和 Flow Matching checkpoint 的张量名称与形状相同，但输出语义不同，因此不能直接混用。使用 DDIM checkpoint 初始化 Flow Matching 时，需要显式设置 `pretrained_ignore_objective=True`；加载逻辑会迁移共享主干并重新初始化输出层。

## 2. 训练方式

### 2.1 全量微调

```python
lora_rank=0
vlm_lora_rank=0
conv_tower="none"
```

此时 `ContextEncoder` 和动作头全部参与训练。DINOv2、SigLIP 等视觉主干仍然冻结，因此这里的“全量”指 E2VLA action expert，而不是把所有视觉 backbone 一起训练。

### 2.2 LoRA

```python
lora_rank=16
conv_tower="none"
```

`lora_rank` 对 `ContextEncoder` 的注意力投影注入 LoRA。动作头仍然全量训练，LayerNorm、bias 和少量 embedding 也保持可训练。

这一路径必须提供 `--pretrained_ckpt`。如果基座权重是随机初始化的，冻结基座后没有可供 LoRA 适配的有效模型，训练程序会直接报错。

`vlm_lora_rank` 是另一套高级选项，它作用于 DINOv2/SigLIP 主干，显存和计算开销明显更高，默认保持 `0`。

### 2.3 ResNet18 可训练旁路

```python
conv_tower="resnet18"
conv_tower_lr_scale=2.0
```

这里的 ResNet18 不是替换视觉 backbone，而是在冻结的 ViT 特征旁边增加一条可训练卷积支路。实现只保留 ResNet18 到 `layer2` 的浅层结构，用于补充边缘、纹理和局部颜色等低层视觉信息。

`conv_tower_lr_scale` 是旁路相对于 `max_lr` 的学习率倍率。旁路可与全量微调或 LoRA 组合；与 LoRA 组合时，LoRA 适配原 ContextEncoder，新增卷积分支仍然全量训练。

## 3. 上下文编码器

`context_encoder` 决定冻结视觉主干与动作头之间如何融合观测信息。

| 值 | 语言 | 相机参数 | 结构定位 |
| --- | --- | --- | --- |
| `vl` | 使用 | 使用 | 原仓库视觉语言编码器，包含 PRoPE 和 QFormer |
| `sa` | 不使用 | 不使用 | 保留与 `vl` 接近的网络容量，用自注意力替代语言交叉注意力 |
| `transformer` | 不使用 | 不使用 | 更小的自注意力编码器，然后池化到 64 个 token |
| `mlp` | 不使用 | 不使用 | 逐 token MLP，然后池化到 64 个 token |

`main` 分支主要使用 `vl`，并通过 LoRA、ResNet18 旁路或全量训练进行对照。`va-no-language` 分支主要使用 `sa`、`transformer` 和 `mlp`，用于单任务、无语言输入的视觉动作实验。

需要注意：切换到 `va-no-language` 分支不会自动修改每个训练预设。是否读取语言最终由 `context_encoder` 决定；要关闭语言，应选择 `va_real_joint_*` 预设或显式设置 `context_encoder`。

无语言编码器不读取相机内外参，因此只能配合关节空间或 `ee_base`，不能配合依赖相机外参的 `ee_cam`。此外，`mlp` 没有注意力投影，不能使用 `lora_rank > 0`。

## 4. 已有训练预设

### 4.1 视觉语言模型

| 配置名 | 动作空间 | 动作算法 | 训练方式 |
| --- | --- | --- | --- |
| `finetune_real` | `ee_cam` | DDIM | LoRA |
| `finetune_real_conv` | `ee_cam` | DDIM | 全量微调 + ResNet18 旁路 |
| `finetune_real_conv_lora` | `ee_cam` | DDIM | LoRA + ResNet18 旁路 |
| `finetune_real_flow` | `ee_cam` | Flow Matching | LoRA |
| `finetune_real_conv_flow` | `ee_cam` | Flow Matching | 全量微调 + ResNet18 旁路 |
| `finetune_real_joint` | `joint7` | DDIM | action expert 全量微调 |
| `finetune_real_joint_flow` | `joint7` | Flow Matching | action expert 全量微调 |
| `finetune_real_joint_conv_flow` | `joint7` | Flow Matching | 全量微调 + ResNet18 旁路 |

前五项使用 HDF5 `RealRobot` 数据和末端位姿；后三项使用当前 memmap `RealBinDataset` 和关节角。

### 4.2 无语言视觉动作模型

| 配置名 | 动作算法 | 编码器 | 训练方式 |
| --- | --- | --- | --- |
| `va_real_joint_sa` | DDIM | `sa` | 无语言，全量训练 |
| `va_real_joint_transformer` | DDIM | `transformer` | 无语言，全量训练 |
| `va_real_joint_mlp` | DDIM | `mlp` | 无语言，全量训练 |
| `va_real_joint_{sa,transformer,mlp}_flow` | Flow Matching | 对应编码器 | 无语言，全量训练 |

这些名称是方便复用的默认组合。单次实验可以直接通过命令行覆盖字段，不需要为每组超参数新增预设。

## 5. 如何修改或新增算法配置

### 5.1 单次实验：使用命令行覆盖

例如，用现有真机关节配置运行 Flow Matching，并将推理步数改为 15：

```bash
python train.py --config finetune_real_joint_flow \
  --inference_timesteps 15 \
  -s JOINT_FLOW_15
```

适合通过命令行覆盖的字段包括 `bs`、`max_lr`、`max_iterations`、日志频率和采样步数。这些改动通常不值得增加新的配置名。

### 5.2 长期复用：在 `CONFIGS` 中注册预设

所有真机关节配置都应从 `make_real_joint_config()` 开始，不要复制整份 `TrainConfig`。

新增一个 DDIM + LoRA 配置：

```python
CONFIGS["joint_ddim_lora"] = make_real_joint_config(
    lora_rank=16,
    max_lr=5e-5,
    max_iterations=int(30e3),
)
```

训练时必须提供预训练权重：

```bash
python train.py --config joint_ddim_lora \
  --pretrained_ckpt PRETRAIN.pt \
  --no-pretrained-strict \
  --pretrained-ignore-action-layout \
  -s JOINT_DDIM_LORA
```

新增一个 Flow Matching 配置时，使用 `_flow_variant()` 派生：

```python
CONFIGS["joint_flow"] = _flow_variant(
    make_real_joint_config(),
    max_iterations=int(60e3),
)
```

新增无语言 SA 配置时，使用 `_va_variant()`，并保持关节动作空间：

```python
CONFIGS["joint_sa"] = _va_variant(
    make_real_joint_config(),
    context_encoder="sa",
    action_space="joint7",
)
```

`_va_variant()` 会清空动作归一化文件路径。应先用新配置计算统计量，再在训练命令中传入生成的文件：

```bash
python -m data_prepare.compute_action_stats \
  --config joint_sa \
  -o ./action_stats/joint_sa.json

python train.py --config joint_sa \
  --action_norm_stats ./action_stats/joint_sa.json \
  -s JOINT_SA
```

新增组合配置时应只覆盖产生差异的字段。例如，Flow Matching + ResNet18：

```python
CONFIGS["joint_flow_resnet"] = _flow_variant(
    make_real_joint_config(),
    conv_tower="resnet18",
    conv_tower_lr_scale=2.0,
)
```

## 6. Checkpoint 与配置兼容性

以下字段会改变模型语义或参数结构，修改后不能把旧 checkpoint 当作同一个训练任务继续：

- `objective`：DDIM 与 Flow Matching 输出语义不同。
- `context_encoder`：`vl`、`sa`、`transformer`、`mlp` 的模块结构不同。
- `action_space`：动作维数和各通道含义可能变化。
- `lora_rank`：会改变 state dict 中注意力层的结构。
- `conv_tower`：会新增卷积分支参数。
- `action_norm_stats`：会改变模型训练和解码动作的数值空间。

`--pretrained_ckpt` 表示开始一次新训练并迁移可兼容权重；`-c` 表示恢复同一个实验。恢复训练时必须使用与原实验相同的算法配置，不能使用 ignore 参数跨越上述差异。

从官方 DDIM、末端位姿 checkpoint 迁移到关节空间 Flow Matching 时，通常需要：

```bash
--no-pretrained-strict \
--pretrained-ignore-action-layout \
--pretrained-ignore-objective
```

这些参数只应在明确知道 checkpoint 与新配置差异的情况下使用。加载日志会列出未迁移或重新初始化的张量，应在正式训练前核对。
