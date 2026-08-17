"""训练参数定义和命名预设。

本项目的主入口是 ``finetune_real_joint``。开发者通常只需要：

1. 在 ``data_utils/dataset_real.py`` 修改数据路径、相机顺序和关节/夹爪定义；
2. 在 ``make_real_joint_config`` 修改项目共享的训练默认值；
3. 用命令行覆盖单次实验参数，例如 ``--max_lr 5e-5``。

不要为一次实验新增预设。只有需要长期复用的配置才加入 ``CONFIGS``，变体应通过
``dataclasses.replace`` 或下面的派生函数创建，避免复制整份配置。
"""

import os
import json
import math
from typing import List, Dict
from dataclasses import dataclass, field, asdict, replace

from data_utils import datasets
from data_utils.dataset_base import H5DatasetMapBase
from train_utils.lora import VLM_LORA_TARGETS, DEFAULT_VLM_LORA_TARGETS
from models.action_expert import OBJECTIVES, FLOW_TIME_SAMPLING
from models.context_encoder import CONTEXT_ENCODERS, DEFAULT_CONTEXT_ENCODER
from models.conv_tower import CONV_TOWERS


@dataclass
class TrainConfig(object):
    """一次训练运行的全部配置。字段顺序即 `-h` 里的顺序，也是 dump 出的 json 里的顺序。"""

    # 模型和初始化
    model: str = "base"  # "tiny" | "small" | "base"

    pretrained_ckpt: str | None = None

    pretrained_weights_only: bool = True

    pretrained_strict: bool = True

    # 参数高效微调和视觉分支
    lora_rank: int = 0

    vlm_lora_rank: int = 0

    vlm_lora_targets: List[str] = field(
        default_factory=lambda: list(DEFAULT_VLM_LORA_TARGETS))

    conv_tower: str = "none"

    conv_tower_lr_scale: float = 1.0

    context_encoder: str = DEFAULT_CONTEXT_ENCODER

    # 动作表示和生成目标
    action_norm_stats: str | None = None

    legacy_gripper_scale: float = 1.0

    action_space: str = "ee_cam"

    objective: str = "ddim"

    diffusion_timesteps: int = 100

    inference_timesteps: int = 0

    flow_time_sampling: str = "uniform"
    flow_time_alpha: float = 1.5  # 只在 flow_time_sampling == "beta" 时读

    pretrained_ignore_action_layout: bool = False

    pretrained_ignore_objective: bool = False

    # 优化器
    bs: int = 32           # batch size
    workers: int = 4       # dataloader 的 num_workers
    fp16: bool = True      # 混合精度（fp32 + bfloat16）

    grad_clip: float = -1  # <= 0 关闭梯度裁剪
    max_lr: float = 1e-4   # 峰值学习率（调度是 constant_with_warmup）
    wd: float = 1e-2       # weight decay
    num_warmup: int = int(10e3)  # warmup 步数

    ema_enabled: bool = False
    ema_start: int = int(400e3)
    ema_decay: float = 0.9995

    # 数据
    dataset_classes: List[type[H5DatasetMapBase] | str] = field(default_factory=list)
    dataset_weights: List[float] | None = None  # 长度必须等于 dataset_classes
    sample_multiplex: int = 1

    # 日志
    log_interval: int = 100

    log_sample_interval: int = 0

    log_image_interval: int = 0

    log_hist_interval: int = 0

    # Checkpoint 和训练时长
    save_interval: int = int(100e3)   # 存成 ckpt_{iter}.pt，< 0 关闭
    save_latest_interval: int = 2000  # 存成 ckpt_latest.pt（顺带按 loss 更新 ckpt_best.pt）
    max_iterations: int = int(600e3)

    def __post_init__(self):
        """还原数据集类，并尽早校验枚举值。"""
        for i, D in enumerate(self.dataset_classes):
            if isinstance(D, str):
                self.dataset_classes[i] = getattr(datasets, D)
            else:
                assert issubclass(D, H5DatasetMapBase)

        if self.objective not in OBJECTIVES:
            raise ValueError(
                "unknown objective '{}'; valid choices are {}"
                .format(self.objective, list(OBJECTIVES)))
        if self.flow_time_sampling not in FLOW_TIME_SAMPLING:
            raise ValueError(
                "unknown flow_time_sampling '{}'; valid choices are {}"
                .format(self.flow_time_sampling, list(FLOW_TIME_SAMPLING)))

        unknown = [t for t in self.vlm_lora_targets if t not in VLM_LORA_TARGETS]
        if unknown:
            raise ValueError(
                "unknown vlm_lora_targets {}; valid targets are {}"
                .format(unknown, sorted(VLM_LORA_TARGETS)))
        if self.vlm_lora_rank > 0 and not self.vlm_lora_targets:
            raise ValueError(
                "vlm_lora_rank={} but vlm_lora_targets is empty, which would adapt "
                "nothing. Set targets or set the rank back to 0."
                .format(self.vlm_lora_rank))

        if self.context_encoder not in CONTEXT_ENCODERS:
            raise ValueError(
                "unknown context_encoder '{}'; valid choices are {}"
                .format(self.context_encoder, list(CONTEXT_ENCODERS)))

        if self.conv_tower not in CONV_TOWERS:
            raise ValueError(
                "unknown conv_tower '{}'; valid choices are {}"
                .format(self.conv_tower, list(CONV_TOWERS)))
        if self.conv_tower_lr_scale <= 0:
            raise ValueError(
                "conv_tower_lr_scale must be positive, got {}. A scale of 0 would put the "
                "branch in the optimizer and never move it, which looks exactly like "
                "training it."
                .format(self.conv_tower_lr_scale))
        if (not math.isfinite(self.legacy_gripper_scale)
                or self.legacy_gripper_scale <= 0):
            raise ValueError(
                "legacy_gripper_scale must be finite and positive, got {}"
                .format(self.legacy_gripper_scale))

    def model_kwargs(self) -> Dict:
        """返回训练和推理共同使用的模型构造参数。"""
        return dict(
            objective=self.objective,
            diffusion_timesteps=self.diffusion_timesteps,
            inference_timesteps=(self.inference_timesteps or None),
            flow_time_sampling=self.flow_time_sampling,
            flow_time_alpha=self.flow_time_alpha,
            conv_tower=self.conv_tower,
            context_encoder=self.context_encoder,
            legacy_gripper_scale=self.legacy_gripper_scale,
        )

    def to_json(self) -> str:
        """序列化配置，并用数据集类名替代 Python 类对象。"""
        items = asdict(self)
        dataset_classes = items["dataset_classes"]
        for i, D in enumerate(dataset_classes):
            if issubclass(D, H5DatasetMapBase):
                dataset_classes[i] = D.__name__
            else:
                assert isinstance(D, str)
        return json.dumps(items, ensure_ascii=False, indent=4)

    def dump(self, path: str):
        save_folder = os.path.dirname(path)
        os.makedirs(save_folder, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            fp.write(self.to_json())

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as fp:
            items = json.load(fp)
        return cls(**items)

def _va_variant(base: TrainConfig, context_encoder: str,
                action_space: str = "ee_base", **overrides) -> TrainConfig:
    """创建无语言、无相机参数的视觉-动作变体。"""
    return replace(
        base,
        context_encoder=context_encoder,
        action_space=action_space,
        action_norm_stats=None,
        lora_rank=0,
        pretrained_ckpt=None,
        dataset_classes=list(base.dataset_classes),
        dataset_weights=(None if base.dataset_weights is None
                         else list(base.dataset_weights)),
        vlm_lora_targets=list(base.vlm_lora_targets),
        **overrides
    )

def _flow_variant(base: TrainConfig, **overrides) -> TrainConfig:
    """创建只将生成目标改为 flow matching 的变体。"""
    return replace(
        base,
        objective="flow",
        dataset_classes=list(base.dataset_classes),
        dataset_weights=(None if base.dataset_weights is None
                         else list(base.dataset_weights)),
        vlm_lora_targets=list(base.vlm_lora_targets),
        **overrides
    )


def make_real_joint_config(**overrides) -> TrainConfig:
    """单机器人、单任务、绝对关节角训练的项目默认配置。

    数据格式相关参数不放在这里；请在 ``RealBinDataset`` 中修改。传入的 ``overrides``
    只用于创建长期复用的派生预设，单次实验优先使用命令行覆盖。
    """
    values = dict(
        dataset_classes=[datasets.RealBinDataset],
        dataset_weights=[1],
        sample_multiplex=1000,
        action_space="joint7",
        action_norm_stats="./action_stats/real_joint7_raw_gripper.json",
        lora_rank=0,
        bs=16,
        max_lr=1e-4,
        grad_clip=1.0,
        num_warmup=int(2e3),
        ema_enabled=True,
        ema_start=int(2e3),
        save_interval=int(5e3),
        save_latest_interval=1000,
        max_iterations=int(60e3),
    )
    values.update(overrides)
    return TrainConfig(**values)


CONFIGS: Dict[str, TrainConfig] = {}

CONFIGS["debug"] = TrainConfig()

# 原仓库预训练与仿真预设；保留用于复现实验。
CONFIGS["pretrain"] = TrainConfig(
    dataset_classes=[
        datasets.Droid,
        datasets.Maniskill,
        datasets.MetaWorld,
    ],
    dataset_weights=[10, 1, 1]
)
CONFIGS["pretrain_extra"] = TrainConfig(
    dataset_classes=[
        datasets.Droid,
        datasets.Maniskill,
        datasets.MetaWorld,
        datasets.PickPlaceCan,
        datasets.OpenDrawer,
        datasets.OpenOven,
    ],
    dataset_weights=[10, 1, 1, 1, 1, 1]
)

CONFIGS["finetune_libero_spatial"] = TrainConfig(
    dataset_classes=[datasets.LiberoSpatial],
    dataset_weights=[1],
    sample_multiplex=1000,
    num_warmup=int(2e3),
    save_interval=int(10e3),
    max_iterations=int(70e3),
    ema_start=int(2e3),
)
CONFIGS["finetune_libero_object"] = TrainConfig(
    dataset_classes=[datasets.LiberoObject],
    dataset_weights=[1],
    sample_multiplex=1000,
    num_warmup=int(2e3),
    save_interval=int(10e3),
    max_iterations=int(70e3),
    ema_start=int(2e3),
)
CONFIGS["finetune_libero_goal"] = TrainConfig(
    dataset_classes=[datasets.LiberoGoal],
    dataset_weights=[1],
    sample_multiplex=1000,
    num_warmup=int(2e3),
    save_interval=int(10e3),
    max_iterations=int(70e3),
    ema_start=int(2e3),
)
CONFIGS["finetune_libero_10"] = TrainConfig(
    dataset_classes=[datasets.Libero10],
    dataset_weights=[1],
    sample_multiplex=1000,
    num_warmup=int(2e3),
    save_interval=int(10e3),
    max_iterations=int(70e3),
    ema_start=int(2e3),
)

# 末端位姿真机预设；当前关节角项目通常不需要修改。
CONFIGS["finetune_real"] = TrainConfig(
    dataset_classes=[datasets.RealRobot],
    dataset_weights=[1],
    sample_multiplex=1000,
    lora_rank=16,
    bs=16,
    max_lr=5e-5,
    grad_clip=1.0,
    num_warmup=int(1e3),
    ema_enabled=True,
    ema_start=int(1e3),
    save_interval=int(5e3),
    save_latest_interval=1000,
    max_iterations=int(20e3),
)

CONFIGS["finetune_real_conv"] = replace(
    CONFIGS["finetune_real"],
    dataset_classes=[datasets.RealRobot],
    dataset_weights=[1],
    vlm_lora_targets=list(DEFAULT_VLM_LORA_TARGETS),
    lora_rank=0,
    conv_tower="resnet18",
    conv_tower_lr_scale=2.0,
)

CONFIGS["finetune_real_conv_lora"] = replace(
    CONFIGS["finetune_real"],
    dataset_classes=[datasets.RealRobot],
    dataset_weights=[1],
    vlm_lora_targets=list(DEFAULT_VLM_LORA_TARGETS),
    lora_rank=16,
    conv_tower="resnet18",
    conv_tower_lr_scale=2.0,
)

# 当前项目的主要训练入口。
CONFIGS["finetune_real_joint"] = make_real_joint_config()

# 生成目标和视觉编码器变体都从基础预设派生。
CONFIGS["pretrain_flow"] = _flow_variant(CONFIGS["pretrain"])
CONFIGS["finetune_libero_10_flow"] = _flow_variant(CONFIGS["finetune_libero_10"])
CONFIGS["finetune_real_flow"] = _flow_variant(CONFIGS["finetune_real"])
CONFIGS["finetune_real_joint_flow"] = _flow_variant(CONFIGS["finetune_real_joint"])

CONFIGS["finetune_real_conv_flow"] = _flow_variant(
    CONFIGS["finetune_real_conv"],
    conv_tower_lr_scale=1.0,
    max_lr=1e-4,
    num_warmup=int(2e3),
    ema_start=int(2e3),
    max_iterations=int(60e3),
)

CONFIGS["finetune_real_joint_conv_flow"] = _flow_variant(
    CONFIGS["finetune_real_joint"],
    conv_tower="resnet18",
    conv_tower_lr_scale=2.0,
    pretrained_strict=False,
    pretrained_ignore_action_layout=True,
    pretrained_ignore_objective=True,
    max_lr=5e-5,
    num_warmup=int(1e3),
    ema_start=int(1e3),
    max_iterations=int(30e3),
)

_JOINT_BASE = CONFIGS["finetune_real_joint"]
for _enc in ("sa", "transformer", "mlp"):
    for _name, _cfg in (
        ("va_libero_10_" + _enc,
         _va_variant(CONFIGS["finetune_libero_10"], _enc)),
        ("va_real_" + _enc,
         _va_variant(CONFIGS["finetune_real"], _enc,
                     max_lr=1e-4,
                     num_warmup=int(2e3),
                     ema_start=int(2e3),
                     max_iterations=int(60e3))),
        ("va_real_joint_" + _enc,
         _va_variant(_JOINT_BASE, _enc, action_space=_JOINT_BASE.action_space)),
    ):
        CONFIGS[_name] = _cfg
        CONFIGS[_name + "_flow"] = _flow_variant(_cfg)
del _JOINT_BASE, _enc, _name, _cfg
