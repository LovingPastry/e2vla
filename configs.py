import os
import json
from typing import List, Dict
from dataclasses import dataclass, field, asdict

from data_utils import datasets
from data_utils.dataset_base import H5DatasetMapBase


@dataclass
class TrainConfig(object):

    model: str = "base"  # choices are ["tiny", "small", "base"]
    pretrained_ckpt: str | None = None  # ckpt path of pretrained model
    # Load only `weights` from pretrained_ckpt, not the optimizer/scaler/EMA state.
    # This is what you want for fine-tuning: a new run re-warms the LR from zero, so
    # inheriting the pretrain run's Adam moments makes the first updates too large.
    # Set False only to imitate the old behaviour. Ignored when resuming with `-c`,
    # which always restores the full optimiser state by design.
    pretrained_weights_only: bool = True
    # Require pretrained_ckpt to match the model exactly. Set False to load only the
    # compatible subset -- needed for checkpoints from before upstream's `compact model`
    # commit, whose ContextEncoder is wider (122.4M vs 102.3M).
    pretrained_strict: bool = True

    # LoRA rank applied to ContextEncoder's attention projections. 0 disables it and
    # trains the whole action expert densely (the historical default). When > 0, the
    # ContextEncoder's base weights are frozen and only the LoRA factors, LayerNorm
    # affines, biases and the two small learned embeddings are trained; the
    # DiffusionHead always stays fully trainable. Intended for small fine-tuning sets,
    # where ContextEncoder (60% of the params, and the part that learns multi-view
    # geometric fusion) is the most over-parameterised half.
    # NOTE: this changes the checkpoint's state_dict layout -- see train_utils/lora.py.
    lora_rank: int = 0

    # Path to a q01/q99 action-statistics JSON, produced by
    #   python -m data_prepare.compute_action_stats --config THIS_CONFIG -o PATH
    # None disables action normalization, which is the historical behaviour: the model
    # then denoises raw camera-relative deltas (metres for translation, near-identity 6D
    # rotation), whose per-channel scales are wildly different from the unit-variance
    # noise DDIM samples.
    #
    # The statistics are over the *model's* action space, not the dataset's raw poses,
    # so a file computed for one dataset/DataConfig combination does not transfer to
    # another. Recompute per fine-tuning set. The resolved stats are copied into every
    # checkpoint, so evaluation does not need this path to still exist.
    action_norm_stats: str | None = None

    # Which action space the head predicts in; see models/action_space.py.
    #   "ee_cam"  -- camera-relative SE(3) delta + openness (10 dim). The default and the
    #                only space the released pretrain checkpoints were trained in.
    #   "jointN"  -- absolute joint angles + openness (N+1 dim), "joint" == "joint7".
    # This is not a knob to flip on an existing run: it changes action_dim, so hist_enc /
    # traj_enc / act_head all change shape and no checkpoint crosses the boundary. It also
    # invalidates action_norm_stats -- the layout stamp in the JSON is checked against it.
    # Joint space additionally drops the DiffusionHead's absolute-position encoding
    # (`abs_pos_enc`), which would need forward kinematics to reconstruct.
    # A pretrained checkpoint from the other space can still be transferred, though --
    # see `pretrained_ignore_action_layout` below; only three layers are bound to the
    # encoding, so "no checkpoint crosses the boundary" applies to an exact load only.
    action_space: str = "ee_cam"

    # Allow `pretrained_ckpt` to come from a DIFFERENT action space. Off by default: the
    # layout stamp exists to stop an accidental cross-space load, which is undetectable
    # once it happens.
    #
    # Turning it on is only sensible for one thing -- transferring the shared trunk. Only
    # three layers are bound to the action encoding (`hist_enc.0`, `traj_enc.0`,
    # `act_head.3`, ~22k params); ContextEncoder's 60.95M and the whole DiT stack in the
    # head are action-space agnostic and keep their exact shapes. Loading an ee_cam
    # checkpoint into a joint run transfers 99.98% of the parameters and re-initialises
    # those three boundary layers.
    #
    # Requires `pretrained_strict=False` (the shapes genuinely do not match), and applies
    # ONLY to `pretrained_ckpt`. Resuming with `-c` always demands an exact match: a
    # resume must continue the same optimisation problem, action space included.
    pretrained_ignore_action_layout: bool = False

    bs: int = 32  # batch size
    workers: int = 4  # num_workers
    fp16: bool = True  # enable mixed precision training (fp32 and bfloat16)

    grad_clip: float = -1  # <= 0 disables the grad clip
    max_lr: float = 1e-4  # maximum learning rate
    wd: float = 1e-2  # weight decay
    num_warmup: int = int(10e3)  # warm up steps

    ema_enabled: bool = False
    ema_start: int = int(400e3)
    ema_decay: float = 0.9995

    dataset_classes: List[type[H5DatasetMapBase] | str] = field(default_factory=list)
    dataset_weights: List[float] | None = None  # len = len(datasets)
    sample_multiplex: int = 1   # set this to a large number (e.g. 1000) if the total number of samples are small

    log_interval: int = 100
    save_interval: int = int(100e3)  # ckpt are named as ckpt_{iter}.pt, set <0 to disable this
    save_latest_interval: int = 2000  # ckpt are named as ckpt_latest.pt
    max_iterations: int = int(600e3)
    
    def __post_init__(self):
        for i, D in enumerate(self.dataset_classes):
            if isinstance(D, str):
                self.dataset_classes[i] = getattr(datasets, D)
            else:
                assert issubclass(D, H5DatasetMapBase)
    
    def dump(self, path: str):
        items = asdict(self)
        dataset_classes = items["dataset_classes"]
        for i, D in enumerate(dataset_classes):
            if issubclass(D, H5DatasetMapBase):
                dataset_classes[i] = D.__name__
            else:
                assert isinstance(D, str)
        
        save_folder = os.path.dirname(path)
        os.makedirs(save_folder, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(items, fp, ensure_ascii=False, indent=4)
    
    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as fp:
            items = json.load(fp)
        return cls(**items)


CONFIGS: Dict[str, TrainConfig] = {}
CONFIGS["debug"] = TrainConfig()
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
    # the global default (400e3) is larger than max_iterations, which would make EMA
    # silently never fire; start it just after warmup instead
    ema_start=int(2e3),
)
CONFIGS["finetune_libero_object"] = TrainConfig(
    dataset_classes=[datasets.LiberoObject],
    dataset_weights=[1],
    sample_multiplex=1000,
    num_warmup=int(2e3),
    save_interval=int(10e3),
    max_iterations=int(70e3),
    # the global default (400e3) is larger than max_iterations, which would make EMA
    # silently never fire; start it just after warmup instead
    ema_start=int(2e3),
)
CONFIGS["finetune_libero_goal"] = TrainConfig(
    dataset_classes=[datasets.LiberoGoal],
    dataset_weights=[1],
    sample_multiplex=1000,
    num_warmup=int(2e3),
    save_interval=int(10e3),
    max_iterations=int(70e3),
    # the global default (400e3) is larger than max_iterations, which would make EMA
    # silently never fire; start it just after warmup instead
    ema_start=int(2e3),
)
CONFIGS["finetune_libero_10"] = TrainConfig(
    dataset_classes=[datasets.Libero10],
    dataset_weights=[1],
    sample_multiplex=1000,
    num_warmup=int(2e3),
    save_interval=int(10e3),
    max_iterations=int(70e3),
    # the global default (400e3) is larger than max_iterations, which would make EMA
    # silently never fire; start it just after warmup instead
    ema_start=int(2e3),
)

# Fine-tuning on a single real robot / single task from a handful of demonstrations
# (order ~50-100 episodes). Differences from the LIBERO presets, all of them motivated
# by the much smaller dataset:
#   - fewer iterations: 70k steps over ~70 episodes is far into the memorisation regime
#   - lower LR: the pretrained action expert only needs adapting, not re-learning
#   - grad clipping on: small batches drawn from few episodes give noisier gradients
#   - EMA on: cheap variance reduction, and it matters most exactly when data is scarce
# Pretrained init via --pretrained_ckpt is recommended but not required: single-task BC
# from scratch on ~50-100 demos is standard practice (Diffusion Policy, ACT). If you
# train from scratch, raise max_iterations and expect more hyperparameter sensitivity.
CONFIGS["finetune_real"] = TrainConfig(
    dataset_classes=[datasets.RealRobot],
    dataset_weights=[1],
    sample_multiplex=1000,
    # LoRA-adapt the ContextEncoder, fully train the DiffusionHead. Set lora_rank=0 to
    # fall back to dense training of all 102M params.
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

# Same real-robot setting as `finetune_real`, but the head predicts absolute joint angles
# instead of camera-relative SE(3) deltas. Trades away two things and buys one:
#   - no absolute-position encoding in the head (needs forward kinematics from joints)
#   - no PRoPE benefit from real extrinsics if the rig is uncalibrated
#   + the output goes straight to the joint controller: no IK, and hand-eye calibration
#     error stops being in the action path at all
#
# A released (ee_cam) checkpoint still transfers: only hist_enc.0 / traj_enc.0 /
# act_head.3 are bound to the action encoding (~22k params), so 99.98% of the weights
# load. It needs both flags below, because the shapes genuinely differ:
#   python train.py --config finetune_real_joint \
#     --pretrained_ckpt PRETRAIN.pt --pretrained_strict False \
#     --pretrained_ignore_action_layout True -s EXP
# max_iterations is sized for training from scratch; halve it when starting from a
# pretrained trunk. With one, lora_rank=16 also becomes usable again -- ContextEncoder
# is then pretrained, which is the precondition LoRA was missing.
CONFIGS["finetune_real_joint"] = TrainConfig(
    dataset_classes=[datasets.RealBinDataset],
    dataset_weights=[1],
    sample_multiplex=1000,
    action_space="joint7",
    # Set both when passing --pretrained_ckpt; see the comment above.
    pretrained_strict=True,
    pretrained_ignore_action_layout=False,
    # Recompute per dataset; the layout stamp in the JSON is checked against action_space.
    #   python -m data_prepare.compute_action_stats --config finetune_real_joint \
    #       -o ./action_stats/real_joint7.json
    action_norm_stats=None,
    lora_rank=0,  # from scratch; raise to 16 if starting from a pretrained trunk
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
