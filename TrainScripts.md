# 1. 在关节空间中训练，默认配置为训练 ContexEncoder + DiT ，动作专家为 **Diffusion Policy**

加载官方开源的预训练模型

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config finetune_real_joint --pretrained_ckpt checkpoints/E2VLA/pretrain/0927_e2vla_base_pretrain_extra/ckpt_0600000.pt --no-pretrained-strict --pretrained-ignore-action-layout  -s JointStates
```

继续训练

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config finetune_real_joint -c JointStates --max_iterations 200000
```

# 2. 在关节空间中训练，默认配置为训练全部e2vla部分参数，动作专家为 **Flow Matching**

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --config finetune_real_joint_flow --pretrained_ckpt checkpoints/E2VLA/pretrain/0927_e2vla_base_pretrain_extra/ckpt_0600000.pt --no-pretrained-strict --pretrained-ignore-action-layout --pretrained-ignore-objective  -s JointFlow
```

继续训练

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --config finetune_real_joint_flow -c JointFlow --max_iterations 300000
```

# 3. 加上ResNet18作为可训练的模块

## 关节空间 + Flow Matching + ResNet18 旁路（全参数训练）

```bash
CUDA_VISIBLE_DEVICES=2 python train.py --config finetune_real_joint_conv_flow \
  --pretrained_ckpt checkpoints/E2VLA/pretrain/0927_e2vla_base_pretrain_extra/ckpt_0600000.pt \
  -s JointConvFlow --num_warmup 1000 --ema_start 1000 --max_iterations 300000
```

## 末端位姿空间的对应版本（对照组）

```bash
CUDA_VISIBLE_DEVICES=2 python train.py --config finetune_real_conv_flow \
  --pretrained_ckpt ./checkpoints/E2VLA/pretrain/0927_e2vla_base_pretrain/ckpt_0600000.pt \
  --pretrained-ignore-objective \
  --max_lr 5e-5 --num_warmup 1000 --ema_start 1000 --max_iterations 300000 \
  -s REAL_CONV_FLOW
```


# 4. VA模型：去掉语言，不用相机内外参

三个上下文编码器：`sa` 57.4M（拓扑不变，语言交叉注意力换成自注意力）、`transformer` 12.4M、`mlp` 4.1M。全部从零训，不能接 `--pretrained_ckpt`（已发布权重是 `vl` 编码器，加载会被拦下）。

## 关节空间（真机 memmap 数据）

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config va_real_joint_sa          -s VA_SA
CUDA_VISIBLE_DEVICES=1 python train.py --config va_real_joint_transformer -s VA_TF
CUDA_VISIBLE_DEVICES=2 python train.py --config va_real_joint_mlp         -s VA_MLP
```

Flow Matching 版本（`_flow` 后缀，除 objective 外与上面完全相同）

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config va_real_joint_sa_flow          -s VA_SA_FLOW
CUDA_VISIBLE_DEVICES=1 python train.py --config va_real_joint_transformer_flow -s VA_TF_FLOW
CUDA_VISIBLE_DEVICES=2 python train.py --config va_real_joint_mlp_flow         -s VA_MLP_FLOW
```

继续训练

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config va_real_joint_sa -c VA_SA --max_iterations 200000
```

## 末端位姿空间（要部署就用这个，关节 checkpoint 只能训不能推）

先把 `RealBinDataset.ACTION_SPACE` 改成 `"ee_base"`

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config va_real_joint_sa --action_space ee_base -s VA_SA_EE
```

## LIBERO-10 对照组

```bash
CUDA_VISIBLE_DEVICES=3 python train.py --config va_libero_10_sa      -s VA_LIBERO_SA
CUDA_VISIBLE_DEVICES=3 python train.py --config va_libero_10_sa_flow -s VA_LIBERO_SA_FLOW
```