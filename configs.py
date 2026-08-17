"""训练配置：`TrainConfig` 的字段定义，以及 `CONFIGS` 预设表。

`python train.py --config NAME` 先从 `CONFIGS` 里取出预设，再由 tyro 接管 `--config`
之后的所有参数去覆盖任意字段（`--max_lr 5e-5`、`--context_encoder mlp`、bool 字段是
裸开关如 `--pretrained_ignore_objective`）。`python train.py -h` 会同时打印 argparse
和 tyro 两份帮助。

这个文件遵守三条约定，读/改之前先知道：

1. **合法取值一律从实现处 import**（`OBJECTIVES` / `CONTEXT_ENCODERS` / `CONV_TOWERS`
   / `VLM_LORA_TARGETS`），不在这里再抄一份。抄出来的列表会漂移，而漂移的后果是
   "配置校验通过、模型里炸掉"，或者更糟——静默走了默认分支。

2. **预设之间靠派生，不手抄**（`_va_variant` / `_flow_variant`）。对照实验只有在"除了
   那一个维度以外全都相同"时才成立，手抄迟早会漏掉一个字段。

3. **有五个字段会写进 checkpoint 并在加载时校验**：`objective`、`context_encoder`、
   `action_space`（存的是它的 layout）、`action_norm_stats`（存的是解析后的统计量）、
   `vlm_lora_rank` / `vlm_lora_targets`。它们不是随手能翻的开关——翻了就是另一个模型，
   而且出问题时张量名和形状全都对得上，只有这几个戳能拦住。见 `train_utils/ckpt.py`。
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

    # ======================================================================
    # 模型规模与权重来源
    # ======================================================================

    model: str = "base"  # "tiny" | "small" | "base"（hdim 768），见 models/vla.py:VLA_SIZES

    # 预训练权重路径。它和 `-c` 续训是两件不同的事：
    #   -c              续上一次被打断的训练，权重/优化器动量/LR 调度位置/迭代数全部恢复
    #   pretrained_ckpt 开一次**新的**训练，只搬权重，见下面 pretrained_weights_only
    pretrained_ckpt: str | None = None

    # 只从 pretrained_ckpt 里取 `weights`，不取 optimizer / scaler / EMA。
    # 微调要的就是这个：新 run 的 LR 从 0 重新 warmup，配上预训练跑到一半的 Adam 动量会让
    # 最初几步的更新过大——小数据集上这恰恰是最不想要的。设 False 只为复现旧行为。
    # `-c` 续训不看这个字段，它按设计总是恢复完整的优化器状态。
    pretrained_weights_only: bool = True

    # 要求 pretrained_ckpt 与当前模型逐张量对齐（名字 + 形状）。
    # 设 False 只加载能对上的子集——只有加载 upstream `compact model`(7eb18ac) 之前的老
    # checkpoint 时才需要（那时的 ContextEncoder 更宽，122.4M vs 102.3M）。
    # 注意代价：它把整份逐张量报告**一次性**放行，包括你没预期的那些不匹配。
    # 只想放行"本次新增的模块"（如 conv 旁路）不需要动它，train.py 走的是按 key 名单放行。
    pretrained_strict: bool = True

    # ======================================================================
    # LoRA：两个 rank，作用域完全不同
    #   lora_rank      -> ContextEncoder（动作专家的一半，本仓库自己的权重）
    #   vlm_lora_rank  -> 冻结的视觉/文本主干（DINOv2 / SigLIP，第三方权重）
    # 详见 README §3.5 与 train_utils/lora.py。
    # ======================================================================

    # 作用于 ContextEncoder 注意力投影的 LoRA 秩。0 = 关闭，全量稠密训练动作专家（历史默认）。
    # > 0 时冻结 ContextEncoder 的基座权重，只训 LoRA 因子 + LayerNorm 仿射 + bias +
    # 两个小 embedding（`qformer.queries` / `main_cam_embed`）；DiffusionHead 永远全量可训。
    # 面向小数据集：ContextEncoder 占 60% 参数，也是最过参数化的那一半。
    # 三个约束：
    #   * 必须配 --pretrained_ckpt。LoRA 分解的是"相对预训练基座的低秩更新"，基座本身
    #     随机初始化的话它什么也适配不了，train.py 会直接报错。
    #   * 它改变 checkpoint 的 state_dict 布局（`...to_q.lin.weight` + `.A`/`.B`）。
    #   * context_encoder="mlp" 没有任何注意力投影可注入，setup_lora 会报错而不是
    #     "注入 0 个然后把整个编码器冻住"。
    lora_rank: int = 0

    # 作用于**冻结主干**（DINOv2 / SigLIP）的 LoRA 秩。0 = 关闭，也是所有已发布权重的
    # 训练方式——整个仓库的设计前提就是主干是固定的特征提取器。
    # 只在真机图像离主干预训练分布很远时才值得开，而且从 configs.py 的预设里开，别从命令行开。
    # 三个代价，按你会撞上的顺序：
    #   1. 两个 ViT 都要为反向保留激活（每样本每相机），步时和显存都显著上升，先把 bs 减半；
    #   2. 任何"冻结特征缓存"都失效——特征不再是图像的固定函数；
    #   3. 因子随 checkpoint 走（`vlm_lora_weights`），推理时重新注入且**故意不 merge**
    #      （bf16 舍入会把这点适配量吃掉）。训练开了、评测忘了开的话，动作专家会零缺失 key
    #      地加载成功，然后读到它从没见过的特征——主干根本不在 state_dict 里，别的都发现不了。
    #      `check_vlm_lora` 就是专门拦这个的。
    vlm_lora_rank: int = 0

    # 适配哪几座塔："dinov2" / "siglip_vision" / "siglip_text" 的任意子集，见
    # train_utils/lora.py:VLM_LORA_TARGETS。默认两座图像塔：单任务数据只有一句 prompt，
    # 适配文本塔等于去拟合那一句话，学不到任何可迁移的东西。vlm_lora_rank == 0 时忽略。
    vlm_lora_targets: List[str] = field(
        default_factory=lambda: list(DEFAULT_VLM_LORA_TARGETS))

    # ======================================================================
    # 视觉旁路：与冻结 ViT 并行的可训练 CNN（models/conv_tower.py，默认关闭）
    # ======================================================================

    # "none" = 原架构，逐位一致：不构建任何模块，state_dict 里不出现任何 key。
    #
    # 它和 vlm_lora_rank 解决同一个问题（真机图像在 DINOv2/SigLIP 分布之外），但便宜得多：
    # 加 ~2.9M 参数和一个浅卷积栈，而不是给两个 ViT 全程录反向图。它读的是原始像素，
    # 因此能学到 patch-16 ViT 丢掉的低层统计——光照、夹爪纹理、没纹理的桌面。
    # 256x256 下步时大约翻倍，激活显存上升；两种方案下冻结主干本身都不动。
    #
    # 它只在配 --pretrained_ckpt 时才有意义：融合层 [I | 0] 初始化让第 0 步与无旁路模型
    # 完全一致，预训练主干精确热启动。从零训就只是多一堆参数。
    # 新增的张量允许在预训练 checkpoint 里缺失，而且**不需要**放松 pretrained_strict——
    # train.py 按 key 名单（CONV_BRANCH_KEYS）单独放行它们。
    conv_tower: str = "none"

    # 卷积旁路的学习率倍率（相对 max_lr）。主干是热启动的、要小步；旁路是 ImageNet 初始化
    # 落到新域、要大步。只影响 optimizer 不影响模型，所以刻意不进 model_kwargs()。
    # conv_tower == "none" 时忽略。
    conv_tower_lr_scale: float = 1.0

    # ======================================================================
    # 上下文编码器：视觉-语言 / 视觉-动作的总开关（models/context_encoder.py）
    # ======================================================================

    # 冻结主干与扩散头之间那一段用哪个编码器：
    #
    #   取值            语言   相机参数   参数量    结构
    #   ------------   ----   --------   -------   --------------------------------
    #   "vl"           有     有(PRoPE)  60.95M    原始架构，已发布权重都是它
    #   "sa"           无     无         57.40M    拓扑不变，语言交叉注意力 -> 第二次自注意力
    #   "transformer"  无     无         12.42M    单个自注意力栈 + 平均池化到 64 token
    #   "mlp"          无     无          4.14M    逐 token MLP + 池化，没有注意力
    #
    # 三个视觉-动作变体**既不读内参也不读外参**，这会约束 action_space：没有外参就没有
    # 相机坐标系可以表达动作，所以它们只接受 "ee_base"（同样的 SE(3) 增量编码，改到机器人
    # 自己的坐标系）或关节空间。`ActionExpert.__init__` 会拒绝 "ee_cam"。
    #
    # 不是能中途翻的开关。"sa" vs "vl" 是容量对齐的语言消融；"transformer"/"mlp" vs "sa"
    # 是模态对齐的容量消融。每个 checkpoint 都会盖上所用的值（`check_context_encoder`）。
    context_encoder: str = DEFAULT_CONTEXT_ENCODER

    # ======================================================================
    # 动作：归一化统计量 + 动作空间（README §3.6 / §3.7）
    # ======================================================================

    # q01/q99 动作统计量 json 的路径，由下面这条命令生成：
    #   python -m data_prepare.compute_action_stats --config 本预设名 -o 路径
    # None = 不做归一化，也是历史行为：此时头部直接去噪原始的相机相对增量（平移是米、旋转是
    # 接近单位阵的 6D），各通道尺度与 DDIM 采样的单位方差噪声差得很远。
    #
    # 统计量是在**模型的**动作空间上算的，不是磁盘上那份数据的属性，所以它同时依赖
    # action_space 和 DataConfig（哪几个相机、采样间隔、预测长度）。换微调集要重算，
    # 换动作空间更要重算。解析后的统计量会拷进每个 checkpoint，所以评测时这个路径可以不存在了。
    action_norm_stats: str | None = None

    # 仅用于兼容夹爪物理量程配置错误的旧 checkpoint。模型完成 q01/q99 反归一化并把
    # gripper action 还原成 [0,1] openness 后，再将最终输出乘这个倍率并 clip 到 [0,1]。
    # 新训练和量程正确的模型必须保持 1.0；旧 real_joint 模型若以 1.5 为满量程、真实满量程
    # 是 0.4314，则在 checkpoint 同目录的配置 JSON 中设为 1.5 / 0.4314。
    legacy_gripper_scale: float = 1.0

    # 头部在哪个空间里预测；见 models/action_space.py。
    #   "ee_cam"  -- 相机相对的 SE(3) 增量 + 夹爪（10 维）。默认值，也是已发布预训练权重
    #                唯一训过的空间。需要外参。
    #   "ee_base" -- 同样的 10 维编码，但表达在机器人自己的（世界/基座）坐标系里。不需要
    #                外参，所以它是视觉-动作变体用的那个。
    #   "jointN"  -- 绝对关节角 + 夹爪（N+1 维），"joint" == "joint7"。天然不需要相机参数。
    #
    # 它改变 action_dim，于是 hist_enc / traj_enc / act_head 三层的形状都变，checkpoint
    # 无法精确跨空间加载；它同时让 action_norm_stats 失效（json 里的 layout 戳会被校验）。
    # 关节空间还会额外去掉头部的绝对位置编码 `abs_pos_enc`（从关节角复原末端位置需要正运动学）。
    # 但"无法跨越"只是指精确加载：只有三层绑定动作编码，主干照样能迁移，见
    # pretrained_ignore_action_layout。
    action_space: str = "ee_cam"

    # ======================================================================
    # 生成目标：DDIM 还是 flow matching（README §3.8）
    # ======================================================================

    # 见 models/action_expert.py:OBJECTIVES。
    #   "ddim" -- DDIM epsilon 预测，训练 diffusion_timesteps 步。默认，已发布权重都是它。
    #   "flow" -- 整流流（最优传输）匹配：头部回归"噪声到干净动作块"这条直线上的速度场，
    #             采样是普通 Euler 积分。推理所需的网络前向次数通常是 DDIM 的一半，这也是
    #             想换它的理由。
    #
    # 不是能中途翻的开关，而且是所有不匹配里下游最发现不了的一个：两种目标产生**逐字节相同**
    # 的 state_dict 布局。每个 checkpoint 都盖了戳，`check_objective` 在不匹配时直接报错。
    # 主干仍然可以迁移，见 pretrained_ignore_objective。
    objective: str = "ddim"

    # DDIM 的训练时间步数。flow 下训练期不离散化任何东西，但这个值仍然定义了头部正弦时间
    # 编码的数值范围（见 `ActionExpert.head_time`），所以除非你清楚在干什么否则别动它——
    # 尤其不要在预训练和它的微调之间改。
    diffusion_timesteps: int = 100

    # 推理时的网络前向次数。0 = "按 objective 的默认值"：DDIM 20，flow 10。
    inference_timesteps: int = 0

    # 训练时 flow 时间 t ∈ [0,1] 的采样方式（t=0 是噪声，t=1 是数据）。只在
    # objective == "flow" 时读；各选项的效果见 `ActionExpert.sample_time`。
    #   "uniform"     -- 整流流原版，没有超参。默认。
    #   "logitnormal" -- sigmoid(N(0,1))，SD3 的选择，权重压在路径中段。
    #   "beta"        -- 1 - Beta(flow_time_alpha, 1)，pi0 的选择，权重压在高噪声端——
    #                    那一端的误差会被后面每一个 Euler 步继承。
    flow_time_sampling: str = "uniform"
    flow_time_alpha: float = 1.5  # 只在 flow_time_sampling == "beta" 时读

    # ======================================================================
    # 两道"我确实要跨过去"的开关
    #
    # 它们分别对应上面的 action_space 和 objective：戳的存在是为了拦住**意外**的跨越，
    # 而这两个 flag 是**故意**跨越时的唯一出口。两者都只作用于 pretrained_ckpt；
    # `-c` 续训永远要求精确匹配（续训必须是同一个优化问题）。
    # ======================================================================

    # 允许 pretrained_ckpt 来自**另一个动作空间**。
    # 唯一合理的用途是迁移共享主干：只有三层绑定动作编码（`hist_enc.0` / `traj_enc.0` /
    # `act_head.3`，约 22k 参数），ContextEncoder 的 60.95M 和头部整个 DiT 栈都与动作空间
    # 无关、形状不变。把 ee_cam 的权重装进关节 run，99.98% 的参数能迁移，那三层重新初始化。
    # 需要同时设 pretrained_strict=False（形状是真的对不上）。
    pretrained_ignore_action_layout: bool = False

    # 允许 pretrained_ckpt 来自**另一个生成目标**——实践中就是用已发布的 DDIM 预训练权重
    # 初始化一个 flow run，这也是最接近"DDIM -> flow 转换"的东西。
    # 动作专家 102.3M 参数里 96.5% 不在乎是哪个目标训出来的：
    #     context_encoder            60.95M (59.6%)  冻结特征 -> 上下文
    #     dp_head.traj_context_attn  37.81M (37.0%)  头部的 DiT 主干
    #     hist_enc / traj_enc / abs_pos_enc / traj_time_embed / denoising_time_embed
    # 这些都在**编码**某样东西（动作块、历史、时间、位置），而那些输入在两种目标下活在同一个
    # 空间里。只有读出层的含义随目标改变，所以 train.py 在迁移后把 `act_head` 的输出 Linear
    # 重新置零——正是从零初始化会留下的状态。epsilon 头约等于速度头的取反，把那一层搬过来
    # 比不搬更糟。`ActionExpert.head_time` 是其余部分真正可复用的前提：它在两种目标下都喂给
    # 头部一个"噪声比例"，预训练的时间条件不会一上来就是反的。
    #
    # 与上面那个不同，它**不**需要 pretrained_strict=False——每个张量都按名字和形状对得上，
    # 这恰恰是危险所在。
    #   python train.py --config finetune_libero_10_flow \
    #     --pretrained_ckpt PRETRAIN_DDIM.pt --pretrained_ignore_objective -s EXP
    pretrained_ignore_objective: bool = False

    # ======================================================================
    # 优化
    # ======================================================================

    bs: int = 32           # batch size
    workers: int = 4       # dataloader 的 num_workers
    fp16: bool = True      # 混合精度（fp32 + bfloat16）

    grad_clip: float = -1  # <= 0 关闭梯度裁剪
    max_lr: float = 1e-4   # 峰值学习率（调度是 constant_with_warmup）
    wd: float = 1e-2       # weight decay
    num_warmup: int = int(10e3)  # warmup 步数

    # EMA 权重。注意 ema_start 必须小于 max_iterations，否则 ema.update() 一次都不会被调用，
    # 而 save_model 照样会写出一个 "ema" 条目——评测时 `--ema` 就会加载**初始**权重，
    # 把整个 run 悄悄丢掉。train.py 里有断言拦这个。
    ema_enabled: bool = False
    ema_start: int = int(400e3)
    ema_decay: float = 0.9995

    # ======================================================================
    # 数据
    # ======================================================================

    # 数据集类（data_utils/datasets.py 里一个类一个数据集）。json 往返时存的是**类名**，
    # `__post_init__` 会把字符串还原成类；`TrajPlanner.set_config` 认的也是这个名字。
    dataset_classes: List[type[H5DatasetMapBase] | str] = field(default_factory=list)
    dataset_weights: List[float] | None = None  # 长度必须等于 dataset_classes
    # 样本总数很少时（单任务几十条轨迹）调大它，例如 1000；否则一个 epoch 走不了几步，
    # 而每个 epoch 边界都要重建 dataloader。必须与 dataset_weights 一起用。
    sample_multiplex: int = 1

    # ======================================================================
    # 日志（TensorBoard，README §3.9）
    #
    # 除了下面三个 log_*_interval 显式标注"默认关闭"的，其余都是默认开启且开销可忽略的。
    # ======================================================================

    # 平均标量的写出频率。便宜的东西——各项 loss、`diag/*` 模型诊断、梯度范数、吞吐——
    # 都按这个频率出，没有理由调大。它同时也是控制台那行 [INFO] 的频率和平均值清零的边界。
    log_interval: int = 100

    # 在当前 batch 上跑**完整采样循环**，按物理单位报告动作误差（`sample/pos_err_m`、
    # `sample/rot_err_deg` …）。这是训练期唯一与 rollout 成功率同量纲的数：训练 loss 量的是
    # 单步去噪，它可以一直降而积分出来的轨迹并没有变好。代价是一次额外前向加
    # inference_timesteps 次头部前向，所以单独限频；500-1000 实测很便宜。0 = 关闭。
    #
    # 它测的是训练 batch（本仓库没有验证集），所以读作"采样器能不能复现它训过的东西"，
    # 而不是泛化。它也是唯一跨 objective / 跨编码器可比的训练期指标。
    log_sample_interval: int = 0

    # 把 batch 的输入图像（最新一帧的所有相机）和 prompt 写进 TensorBoard。任何新数据集的
    # 前几千步都值得开：这是检查 /255 缩放、相机**顺序**（0 号相机定义整个动作坐标系）和
    # prompt 文本是否符合预期的最便宜手段。0 = 关闭。
    # 视觉-动作模型下 prompt 不会被记录——模型不读它，记了会被误读成"指令起作用了"。
    log_image_interval: int = 0

    # 逐张量的权重/梯度直方图。默认关闭——这是这里唯一开销可见的日志开关（步时和事件文件
    # 大小都涨），而它回答的问题 `gnorm/*` 那组标量通常已经回答了。怀疑某层饱和或死掉时再开。
    log_hist_interval: int = 0

    # ======================================================================
    # 保存
    # ======================================================================

    save_interval: int = int(100e3)   # 存成 ckpt_{iter}.pt，< 0 关闭
    save_latest_interval: int = 2000  # 存成 ckpt_latest.pt（顺带按 loss 更新 ckpt_best.pt）
    max_iterations: int = int(600e3)

    def __post_init__(self):
        """字符串 -> 类的还原，加上四组取值校验。

        校验放在这里而不是用到的地方：一个拼错的 target 否则要等 dataloader 和两个主干都起来
        之后才会暴露出来。
        """
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
        """本配置里 `models/vla.py:vla_*` 需要的那个子集。

        放在这里是为了让训练和评测不会漂移：`infer_utils/planner.py` 是从 run 存下来的
        config json 重建模型的，一个只有 train.py 知道的字段到了评测时就会静默回落到默认值。

        `action_norm` / `action_space` 刻意不在这里——它们从 checkpoint 本身解析，
        checkpoint 的优先级高于 json。
        """
        return dict(
            objective=self.objective,
            diffusion_timesteps=self.diffusion_timesteps,
            # 0 是"让 objective 决定"在配置层的写法，模型构造函数那边把同一件事写成 None
            inference_timesteps=(self.inference_timesteps or None),
            flow_time_sampling=self.flow_time_sampling,
            flow_time_alpha=self.flow_time_alpha,
            # 必须在这里，不能只写在 train.py：旁路的张量**在** checkpoint 里，评测时如果
            # 没重建这个分支就会加载失败——报错很响，但要等主干和仿真器都起来之后。
            # 老的 config json 没有这个 key，回落到 "none"。
            conv_tower=self.conv_tower,
            # 同理，再高一层：它决定构造哪个 ContextEncoder 类。评测端无法从 checkpoint 的
            # 张量里反推出来（只能猜）。老 json 回落到 "vl"。
            context_encoder=self.context_encoder,
            legacy_gripper_scale=self.legacy_gripper_scale,
        )

    def to_json(self) -> str:
        """`dump` 写出的确切文本。

        单独拆出来是为了让 TensorBoard 里那份配置和 checkpoint 旁边那份走同一条序列化路径——
        看曲线时旁边显示的配置，必须就是 checkpoint 携带的那份。
        """
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


# ==========================================================================
# 派生函数：预设之间靠它们生成，不要手抄
#
# 两个函数都必须重新包装三个可变字段（dataset_classes / dataset_weights /
# vlm_lora_targets）。`dataclasses.replace` 拷贝的是字段的**值**，不重新包的话派生出来的
# 配置会和母配置共用同一个 list 对象，`__post_init__` 里那句原地改写就会在同一个 list 上
# 跑两次。
#
# 两者可以叠加，`_flow_variant(_va_variant(base, "sa"))` 就是"视觉-动作 + flow"。
# ==========================================================================


def _va_variant(base: TrainConfig, context_encoder: str,
                action_space: str = "ee_base", **overrides) -> TrainConfig:
    """`base` 的视觉-动作副本：没有语言，没有相机内外参。

    三个字段必须一起改，缺一不可：

    * `context_encoder` 选编码器，同时也就砍掉了 SigLIP 文本塔和两处标定的使用
      （见 models/context_encoder.py）。
    * `action_space` 必须离开 "ee_cam"——相机相对的动作需要这个模型不再接收的外参，
      `ActionExpert.__init__` 会明确拒绝。默认换成 "ee_base"（同一套 t3r6 + 夹爪编码，
      改到机器人自己的坐标系）。关节空间本来就不需要相机参数，所以关节预设把自己的
      action_space 原样传进来即可。
    * `action_norm_stats` 清空：统计量是在**模型的**动作空间上算的，换了参考系每个平移
      通道都会变。用 `python -m data_prepare.compute_action_stats --config 新预设名` 重算。
      （关节预设的统计量其实不受编码器影响，这里一并清空只是因为母预设本来也没设。）

    `lora_rank` 和 `pretrained_ckpt` 也清空，理由同源：没有任何已发布 checkpoint 含有这些
    张量，而 LoRA 在随机初始化的主干上什么也分解不出来。
    """
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
    """`base` 换成 flow 目标训练的副本，其余一律不动。

    `objective` 与网络结构、动作空间、数据都是正交的一维，所以这个函数可以套在任何预设外面，
    包括 `_va_variant` 的产物。
    """
    return replace(
        base,
        objective="flow",
        dataset_classes=list(base.dataset_classes),
        dataset_weights=(None if base.dataset_weights is None
                         else list(base.dataset_weights)),
        vlm_lora_targets=list(base.vlm_lora_targets),
        **overrides
    )


# ==========================================================================
# 预设表
#
# 命名约定：
#   {pretrain|finetune}_{数据集}[_变体…][_flow]   视觉-语言模型（原始架构）
#   va_{数据集}_{编码器}[_flow]                    视觉-动作模型（无语言、无相机参数）
#
# 后缀就是它相对同名基础预设改动的那一维：_conv 加卷积旁路，_lora 用 LoRA，_joint 换关节
# 空间，_flow 换生成目标。想加新组合，套上面两个派生函数，别复制粘贴字段。
# ==========================================================================

CONFIGS: Dict[str, TrainConfig] = {}

# 全默认值，只用来跑通流程（没有数据集，dataloader 是空的）。
CONFIGS["debug"] = TrainConfig()

# --------------------------------------------------------------------------
# 预训练：多数据集混合，权重 10:1:1 让 DROID 主导
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# LIBERO 四个 task suite。四份配置除了数据集类以外完全相同：
#   sample_multiplex=1000  轨迹条数少，不放大的话一个 epoch 走不了几步
#   num_warmup=2e3         微调不需要 1e4 步的 warmup
#   ema_start=2e3          全局默认 400e3 比 max_iterations 还大，EMA 会一次都不更新，
#                          而 checkpoint 里照样有 "ema" 条目——评测加 --ema 就等于加载
#                          初始权重。这里挪到 warmup 刚结束。
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# 真机：单机器人单任务，几十条示教（~50-100）
#
# 相对 LIBERO 预设的每一处改动都由"数据少得多"这一点驱动：
#   - 迭代数更少：70k 步跑 ~70 条轨迹已经深入记忆区
#   - 学习率更低：预训练的动作专家只需要适配，不需要重学
#   - 开梯度裁剪：小 batch 从少数几条轨迹里抽，梯度噪声更大
#   - 开 EMA：便宜的方差削减，而且恰恰在数据稀缺时最有用
#
# 推荐但不强制配 --pretrained_ckpt：单任务 BC 从零训 50-100 条示教是标准做法
# （Diffusion Policy、ACT）。从零训就把 max_iterations 调大，并预期对超参更敏感。
# --------------------------------------------------------------------------
CONFIGS["finetune_real"] = TrainConfig(
    dataset_classes=[datasets.RealRobot],
    dataset_weights=[1],
    sample_multiplex=1000,
    # LoRA 适配 ContextEncoder，全量训练 DiffusionHead。设 0 退回全量训练 102M 参数。
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

# `finetune_real` 的两个卷积旁路版本。它们成对存在是因为哪个更好是个经验问题，而
# `finetune_real` 自己是第三条对照——用**同一个**预训练 checkpoint 把三个都跑一遍比成功率：
#
#   finetune_real           LoRA 主干，无旁路    <- 基线
#   finetune_real_conv      主干全开，有旁路     <- 容量最大
#   finetune_real_conv_lora LoRA 主干，有旁路    <- 最接近常规做法："预训练过的低秩适配，
#                                                  新加的全量训练"
#
# 两个都需要 --pretrained_ckpt。旁路的张量在已发布 checkpoint 里必然缺失，train.py 按 key
# 名单放行，不需要放松 pretrained_strict，其它任何不匹配照样报错。显存不够就把 bs 减半——
# 与冻结 ViT 不同，旁路要在全分辨率上保留反向激活。
#
# conv_tower_lr_scale=2.0 是个起点不是调过的值：主干从 5e-5 热启动，ImageNet 初始化的旁路
# 跑 1e-4。
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

# 同样的真机设定，但头部预测绝对关节角而不是相机相对 SE(3) 增量。用两样换一样：
#   - 头部没有绝对位置编码了（从关节角复原末端位置需要正运动学）
#   - 未标定的机架上，本来也拿不到真外参带来的 PRoPE 收益
#   + 输出直接进关节控制器：不过 IK，手眼标定误差彻底退出动作链路
#
# 已发布的 (ee_cam) checkpoint 仍能迁移：只有 hist_enc.0 / traj_enc.0 / act_head.3 绑定
# 动作编码（约 22k 参数），99.98% 的权重能加载。形状是真的对不上，所以两个 flag 都要给：
#   python train.py --config finetune_real_joint \
#     --pretrained_ckpt PRETRAIN.pt --pretrained_strict False \
#     --pretrained_ignore_action_layout True -s EXP
# max_iterations 是按从零训配的；从预训练主干出发就减半。有预训练主干时 lora_rank=16 也
# 重新可用了——ContextEncoder 那时才有基座可分解。
CONFIGS["finetune_real_joint"] = TrainConfig(
    dataset_classes=[datasets.RealBinDataset],
    dataset_weights=[1],
    sample_multiplex=1000,
    action_space="joint7",
    # 传 --pretrained_ckpt 时把这两个设成 False/True，见上面的注释
    pretrained_strict=True,
    pretrained_ignore_action_layout=False,
    # 按数据集重算；json 里的 layout 戳会与 action_space 校验：
    #   python -m data_prepare.compute_action_stats --config finetune_real_joint \
    #       -o ./action_stats/real_joint7.json
    action_norm_stats="./action_stats/real_joint7.json",
    lora_rank=0,  # 从零训；从预训练主干出发的话可以调到 16
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

# --------------------------------------------------------------------------
# flow 家族：与同名 DDIM 预设除 objective 外完全一致，所以两者的对照在数据、schedule、
# 模型规模上是严格可比的。
#
# 已发布的 (DDIM) 预训练权重默认**装不进**这些配置：两种目标的 state_dict 布局完全相同，
# 加载会静默成功，所以 `check_objective` 直接拒绝。故意跨越的路径是
# `--pretrained_ignore_objective True`（迁移主干并把 `act_head` 输出层重新置零）。
# 另外两条路是先用 `pretrain_flow` 自己预训练，或者调大 max_iterations 从零训微调。
#
# 注意两种 objective 的 loss 数值不可比（权重在两边的性质不同），判断 flow 好不好要看
# rollout，或者看 `sample/*`——那组数是跨 objective 可比的。
# --------------------------------------------------------------------------
CONFIGS["pretrain_flow"] = _flow_variant(CONFIGS["pretrain"])
CONFIGS["finetune_libero_10_flow"] = _flow_variant(CONFIGS["finetune_libero_10"])
CONFIGS["finetune_real_flow"] = _flow_variant(CONFIGS["finetune_real"])
# 关节空间的 flow 版本，无卷积旁路（有旁路的是下面的 finetune_real_joint_conv_flow）。
# TrainScripts.md 第 2 节用的就是它。
CONFIGS["finetune_real_joint_flow"] = _flow_variant(CONFIGS["finetune_real_joint"])

# flow + 可训练卷积旁路 + 全量稠密训练，**从零训**。
#
# 每一处与 `finetune_real_conv` 的差异都源于同一件事：那个预设是按"从预训练 checkpoint
# 热启动"配的，而这个没有任何东西可以热启动。schedule 抄自 `finetune_real_joint`，
# 那是本仓库另一个从零训的真机预设。
#
#   python train.py --config finetune_real_conv_flow -s EXP
#
# conv_tower_lr_scale 退回 1.0 是这里最值得理解的一处：它在 `finetune_real_conv` 里是 2.0，
# 因为那时主干热启动要小步、ImageNet 初始化的旁路要大步。从零训这个不对称就不存在了——
# 主干也是随机的——而且此时旁路反而是全模型里**唯一**带预训练权重的部分，再给它加速就是反的。
# 在有实测说明之前，全模型一个学习率。
#
# flow_time_sampling 保持 "uniform"（整流流自己的选择）。只有在少步采样看起来欠收敛时才去
# 试 "logitnormal" 或 "beta"，见 README §3.8。
CONFIGS["finetune_real_conv_flow"] = _flow_variant(
    CONFIGS["finetune_real_conv"],
    conv_tower_lr_scale=1.0,
    # 从零训的 schedule：finetune_real 的 5e-5 / 1e3 / 20e3 是微调的数
    max_lr=1e-4,
    num_warmup=int(2e3),
    ema_start=int(2e3),
    max_iterations=int(60e3),
)

# 关节空间的对应版本：绝对关节角而不是相机相对 SE(3)，flow 而不是 DDIM，外加可训练卷积旁路，
# 全部稠密训练（lora_rank=0）。
#
#   CUDA_VISIBLE_DEVICES=0 python train.py --config finetune_real_joint_conv_flow \
#     --pretrained_ckpt ./checkpoints/E2VLA/pretrain/0927_e2vla_base_pretrain/ckpt_0600000.pt \
#     -s EXP
#
# 三个 pretrained_* 的值写死在预设里而不是留给命令行，因为这个预设只有热启动才有意义，
# 而那时三个都是必需的（设了 ignore_action_layout 却留着 pretrained_strict=True，train.py
# 会报错）。对着已发布的 (ee_cam, DDIM) 预训练权重实测：104.692M 里有 101.726M (97.2%)
# 能迁移，迁不过来的是
#   - 4 个形状不匹配，0.018M：hist_enc.0 / traj_enc.0 / act_head.3 的 weight+bias，
#     它们绑定动作编码，这边 8 维那边 10 维
#   - 6 个多余 key：`dp_head.abs_pos_enc`，关节空间根本不构建它
#   - 74 个缺失 key，2.951M：本次新增的卷积旁路
#
# 这条路径上 pretrained_strict=False 的代价值得单说，因为这正是本仓库平时极力避免的东西：
# `load_actor_weights` 的 allow_missing_prefixes 通道要求"除了点名的前缀之外报告是干净的"，
# 而跨动作空间本身就不干净。于是卷积旁路那些缺失 key 只能和形状不匹配一起被放行，而不是被
# 单独点名。这条路上没法避免——请读它打印的逐张量报告，确认不匹配的就是上面那四个。
#
# 警告：关节 checkpoint 是训练专用的。`infer_utils/planner.py` 无条件按 17 维 SE(3) 解码，
# 所以它在 remote_service 里能正常构建，然后在 `reshape(..., 4, 4)` 崩掉。见 CLAUDE.md。
CONFIGS["finetune_real_joint_conv_flow"] = _flow_variant(
    CONFIGS["finetune_real_joint"],
    conv_tower="resnet18",
    # 回到 2.0，与上面的 finetune_real_conv_flow 相反：那里主干也是随机的，没有"比谁快"的
    # 对象；这里主干是热启动的、要小步，而 ImageNet 初始化的旁路要大步。
    conv_tower_lr_scale=2.0,
    # 这条路径的必需项，见上面的整段说明
    pretrained_strict=False,
    pretrained_ignore_action_layout=True,
    pretrained_ignore_objective=True,
    # finetune_real_joint 的 1e-4 / 2e3 / 60e3 是按从零训配的，它自己的注释也说了从预训练
    # 主干出发要减半。下面这组就是热启动的数；loss 还在动就把 max_iterations 调回去。
    max_lr=5e-5,
    num_warmup=int(1e3),
    ema_start=int(1e3),
    max_iterations=int(30e3),
)

# --------------------------------------------------------------------------
# 视觉-动作（VA）家族：无语言、无相机内外参
#
# 三个数据集 × 三个编码器 × 两个 objective = 18 个预设，成对生成而不是手写出来，因为每一行
# 里唯一不同的就是编码器——这正是重点：只有数据、schedule、动作空间、objective 全都固定住，
# 对照才成立。
#
#   va_libero_10_{sa,transformer,mlp}       LIBERO-10，动作空间 ee_base
#   va_real_{sa,transformer,mlp}            HDF5 真机（RealRobot），动作空间 ee_base
#   va_real_joint_{sa,transformer,mlp}      memmap 真机（RealBinDataset），动作空间 joint7
#   以上每一个都额外有一个 _flow 后缀的兄弟，除 objective 外与它完全相同
#
# 各行的取舍：
#   * 编码器：sa 57.40M 是容量对齐的语言消融；transformer 12.42M / mlp 4.14M 是模态对齐的
#     容量消融。三者输出都是 64 个 token，扩散头一行代码都不用改。
#   * 动作空间：EE 那两行换 "ee_base"（相机相对需要外参，而这些模型不收外参）；关节那行
#     不用换——绝对关节角从来就没参照过相机坐标系，`AbsJoint.uses_camera_pose` 一直是 False。
#   * schedule：LIBERO 行直接继承 `finetune_libero_10`（本来就是 1e-4 从零训的配置）；
#     EE 真机行要覆盖掉 `finetune_real` 的热启动 schedule（5e-5 / 1e3 / 20e3）；
#     关节行继承 `finetune_real_joint`，那本来就是从零训的配置。
#
# 全部**从零训**：没有任何已发布 checkpoint 含有这些张量，也不可能有——state_dict 布局就不
# 一样，`check_context_encoder` 会拦下来。想要预训练收益就得先用 VA 配置跑一遍 pretrain。
#
# 关节那三行是 memmap 真机数据当前形态（`RealBinDataset.ACTION_SPACE = "joint7"`）直接能跑
# 的；但关节 checkpoint 是训练专用的（planner 按 17 维 SE(3) 解码），要部署就把
# `RealBinDataset.ACTION_SPACE` 改成 "ee_base" 然后用 va_real_* 那一行。
# --------------------------------------------------------------------------
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
