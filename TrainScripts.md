# 在关节空间中训练，默认配置为训练 ContexEncoder + DiT ，动作专家为 Diffusion Policy

加载官方开源的预训练模型

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config finetune_real_joint --pretrained_ckpt checkpoints/E2VLA/pretrain/0927_e2vla_base_pretrain_extra/ckpt_0600000.pt --no-pretrained-strict --pretrained-ignore-action-layout -s JointStates
```

# 加上ResNet18作为可训练的模块

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config finetune_real_conv_flow \
  --pretrained_ckpt ./checkpoints/E2VLA/pretrain/0927_e2vla_base_pretrain/ckpt_0600000.pt \
  --pretrained_ignore_objective True \
  --max_lr 5e-5 --num_warmup 1000 --ema_start 1000 --max_iterations 300000 \
  -s REAL_CONV_FLOW
```
