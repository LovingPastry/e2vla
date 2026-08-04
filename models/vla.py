from typing import Optional
from torch import nn, Tensor

from .vlm import VLM
from .action_expert import ActionExpert, DEFAULT_OBJECTIVE
from .action_norm import ActionNormalizer, build_action_normalizer
from .action_space import ActionSpace, build_action_space


class VLA(nn.Module):
    def __init__(
        self,
        hdim: int,
        num_heads: int,
        num_actor_context_layers: int,
        num_actor_diffusion_layers: int,

        diffusion_timesteps: int = 100,
        # None means "whatever the objective's default is" -- 20 for DDIM (unchanged from
        # when this was hardcoded), 10 for flow. Passing a number overrides both.
        inference_timesteps: Optional[int] = None,
        action_norm: Optional[ActionNormalizer | str | dict] = None,
        action_space: Optional[str | ActionSpace] = None,
        objective: str = DEFAULT_OBJECTIVE,
        flow_time_sampling: str = "uniform",
        flow_time_alpha: float = 1.5,
    ):
        super().__init__()
        self.action_space = build_action_space(action_space)
        self.objective = objective
        self.vlm = VLM()
        self.actor = ActionExpert(
            hdim=hdim,
            num_heads=num_heads,
            num_context_layers=num_actor_context_layers,
            num_diffusion_layers=num_actor_diffusion_layers,
            diffusion_timesteps=diffusion_timesteps,
            inference_timesteps=inference_timesteps,
            action_norm=build_action_normalizer(
                action_norm, expect_layout=self.action_space.layout),
            action_space=self.action_space,
            objective=objective,
            flow_time_sampling=flow_time_sampling,
            flow_time_alpha=flow_time_alpha,
        )
    
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                if m.bias is not None and m.bias.requires_grad:
                    # Do not modify the bias in fronzen backbones!!!
                    nn.init.zeros_(m.bias)
    
    def parameter_groups(self):
        decay = []
        no_decay = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if (
                name.endswith(".bias") or 
                "norm" in name.lower() or 
                "qformer.queries" in name
            ):
                no_decay.append(param)
            else:
                decay.append(param)
        
        return decay, no_decay
    
    def forward(
        self, 
        rgbs: Tensor,
        obs_norm_xys: Tensor,
        obs_extrinsics: Tensor, 
        prompt_text: Optional[Tensor], 

        ee_poses: Tensor, 
        history_actions: Tensor, 
        future_actions: Tensor, 
        valid_ee_mask: Tensor, 
        inference: bool, 
        fp16: bool,
    ):
        """
        Args:
            rgbs: (B, To, ncam, 3, H, W)，观测 RGB 图像
            obs_norm_xys: (B, To, ncam, 2, H, W)，归一化相机平面上的坐标
                * 即逐像素的 ((u,v) - (cx,cy)) / (fx,fy)，等价于把像素反投影到 z=1 平面
                * 与外参一起用于构造 Plücker 射线位置编码
            obs_extrinsics: (B, To, ncam, 4, 4)，相机外参 ^{world}_{camera} T
            prompt_text: (B, Lang, E) 或 None，语言指令

            ee_poses: (B, Nee, 4, 4)，当前末端位姿 ^{world}_{ee} T
            history_actions: (B, nhist, Nee, 4*4+1)，世界坐标系下的历史末端状态
                * 4x4 为展平后的变换矩阵
                * 1 为夹爪开合度，范围 [0（闭合），1（张开）]
            future_actions: (B, Ta, Nee, 4*4+1)，世界坐标系下的真值未来动作
                * 4x4 为展平后的变换矩阵
                * 1 为夹爪开合度，范围 [0（闭合），1（张开）]
                * 注意：当 `inference` 为 True 时，仅用它来推断预测动作的形状
            valid_ee_mask: (B, Nee)，只在这些末端执行器上计算损失
            inference: 为 True 时返回预测轨迹，否则返回损失和用于记录的指标
            fp16: 为 True 时使用 bfloat16

        Returns
        -------
        （当 inference 为 True 时）
            pred_future_actions (Tensor): (B, Ta, Nee, 4*4+1)
                * 4x4 为展平后的变换矩阵
                * 1 为夹爪开合度，范围 [0（闭合），1（张开）]
        （否则）
            loss (Tensor): 标量张量
            metrics (Dict[str, Tensor]): 用于记录的指标
        """
        vl_obs, vl_feature = self.vlm(
            rgbs=rgbs,
            obs_norm_xys=obs_norm_xys,
            obs_extrinsics=obs_extrinsics,

            prompt_text=prompt_text,
            fp16=fp16
        )

        return self.actor(
            vl_obs=vl_obs,
            vl_feature=vl_feature,

            ee_poses=ee_poses,
            history_actions=history_actions,
            future_actions=future_actions,
            valid_ee_mask=valid_ee_mask, 
            inference=inference,
            fp16=fp16
        )


# (hdim, num_heads) per size name. The layer counts are the same across all three.
VLA_SIZES = {
    "tiny": (192, 3),
    "small": (384, 6),
    "base": (768, 12),
}


def _build_vla(
    size: str,
    diffusion_timesteps: int = 100,
    inference_timesteps: Optional[int] = None,
    action_norm: Optional[ActionNormalizer | str | dict] = None,
    action_space: Optional[str | ActionSpace] = None,
    objective: str = DEFAULT_OBJECTIVE,
    flow_time_sampling: str = "uniform",
    flow_time_alpha: float = 1.5,
):
    hdim, num_heads = VLA_SIZES[size]
    return VLA(
        hdim=hdim,
        num_heads=num_heads,
        num_actor_context_layers=8,
        num_actor_diffusion_layers=4,
        diffusion_timesteps=diffusion_timesteps,
        inference_timesteps=inference_timesteps,
        action_norm=action_norm,
        action_space=action_space,
        objective=objective,
        flow_time_sampling=flow_time_sampling,
        flow_time_alpha=flow_time_alpha,
    )


# The three public entry points. `train.py` and `infer_utils/planner.py` both resolve them
# by name (`getattr(vla, "vla_" + cfg.model)`), so they have to exist as real attributes.
def vla_tiny(**kwargs):
    return _build_vla("tiny", **kwargs)


def vla_small(**kwargs):
    return _build_vla("small", **kwargs)


def vla_base(**kwargs):
    return _build_vla("base", **kwargs)



def count_parameters():
    # model = vla_tiny()
    # model = vla_small()
    model = vla_base()
    print(model)

    modules = [
        model
    ]

    num_total = 0
    num_trainable = 0
    for m in modules:
        for p in m.parameters():
            num_total += p.numel()
            if p.requires_grad:
                num_trainable += p.numel()

    print("[INFO] Total {:.3f}M parameters, {:.3f}M frozen, {:.3f}M trainable"
          .format(num_total / 1e6, (num_total - num_trainable) / 1e6, num_trainable / 1e6))


if __name__ == "__main__":
    count_parameters()
