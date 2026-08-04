import os
import json
from typing import List, Dict
from dataclasses import dataclass, field, asdict, replace

from data_utils import datasets
from data_utils.dataset_base import H5DatasetMapBase
from train_utils.lora import VLM_LORA_TARGETS, DEFAULT_VLM_LORA_TARGETS
# The valid values are defined next to the code that implements them, not duplicated here
# -- a second list would drift, and the failure of a drifted list is a config that
# validates and then blows up (or worse, silently falls back) inside the model.
from models.action_expert import OBJECTIVES, FLOW_TIME_SAMPLING


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

    # LoRA rank applied to the FROZEN VLM backbones (DINOv2 / SigLIP). 0 disables it and
    # is the default everywhere -- the whole design of this repo assumes the backbones are
    # frozen feature extractors, and every released checkpoint was trained that way.
    #
    # Turn it on from a config preset, not from the command line, and only for a real
    # fine-tuning set whose images are far from the backbones' pretraining distribution.
    # Three consequences, in the order you will hit them:
    #   - both ViTs now need their activations kept for the backward pass, for every
    #     camera of every sample. Step time and memory rise sharply; halve `bs` first.
    #   - the frozen-feature cache and any exported-encoder fast path stop being valid,
    #     because the features are no longer a fixed function of the image.
    #   - the factors travel in the checkpoint (`vlm_lora_weights`) and are re-injected at
    #     inference, and unlike the action expert's they are never merged into the base
    #     weights (bf16 rounding would swallow them -- see infer_utils/planner.py). A
    #     checkpoint trained with this and evaluated without it loads without a single
    #     missing key -- the backbones are not in the state_dict at all -- and then reads
    #     features the action expert never saw. `check_vlm_lora` makes that mismatch loud.
    vlm_lora_rank: int = 0
    # Which towers to adapt: any subset of "dinov2", "siglip_vision", "siglip_text"
    # (see train_utils/lora.py:VLM_LORA_TARGETS). Defaults to the two image towers --
    # a single-task set carries one prompt string, so adapting the text encoder fits that
    # string rather than anything that transfers. Ignored when vlm_lora_rank == 0.
    vlm_lora_targets: List[str] = field(
        default_factory=lambda: list(DEFAULT_VLM_LORA_TARGETS))

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

    # The generative objective the action head is trained with; see
    # models/action_expert.py:OBJECTIVES.
    #   "ddim" -- DDIM epsilon prediction over `diffusion_timesteps` steps. The default,
    #             and what every released pretrain checkpoint was trained with.
    #   "flow" -- rectified-flow (optimal-transport) matching: the head regresses the
    #             velocity along the straight line between noise and the clean chunk, and
    #             sampling is plain Euler integration. Typically needs half the network
    #             evaluations of DDIM at inference, which is the reason to want it.
    #
    # This is NOT a knob to flip mid-run, and it is the one mismatch nothing downstream
    # could otherwise detect: the two objectives produce byte-identical state_dict
    # layouts. Every checkpoint is stamped with the value used and `check_objective`
    # refuses a mismatch on load. The *trunk* still transfers, though -- see
    # `pretrained_ignore_objective`.
    objective: str = "ddim"

    # Number of DDIM training timesteps. Under "flow" nothing is discretised at training
    # time, but this still sets the numeric range the head's sinusoidal time embedding is
    # defined over (see `ActionExpert.head_time`), so leave it alone unless you know why
    # you are changing it -- and never change it between a pretrain and its fine-tune.
    diffusion_timesteps: int = 100

    # Network evaluations at inference. 0 means "the objective's default": 20 for DDIM
    # (unchanged from when it was hardcoded), 10 for flow.
    inference_timesteps: int = 0

    # How the flow time t in [0,1] is drawn during training (t=0 noise, t=1 data). Only
    # read when objective == "flow"; see `ActionExpert.sample_time` for what each does.
    #   "uniform"     -- the rectified-flow baseline, no knob. Default.
    #   "logitnormal" -- sigmoid(N(0,1)), SD3's choice; weights the middle of the path.
    #   "beta"        -- 1 - Beta(flow_time_alpha, 1), pi0's choice; weights the
    #                    high-noise end, where an error propagates through every
    #                    remaining Euler step.
    flow_time_sampling: str = "uniform"
    flow_time_alpha: float = 1.5  # only read when flow_time_sampling == "beta"

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

    # Allow `pretrained_ckpt` to come from a DIFFERENT generative objective -- in
    # practice, initialising a flow run from a released DDIM pretrain checkpoint. Off by
    # default for the usual reason: the stamp exists to stop this happening by accident,
    # and an accidental cross-objective load is undetectable afterwards.
    #
    # Turning it on deliberately is well founded, and it is the closest thing there is to
    # a DDIM -> flow conversion. Of the action expert's 102.3M parameters, 96.5% do not
    # care which objective trained them:
    #     context_encoder            60.95M (59.6%)  frozen VLM features -> context
    #     dp_head.traj_context_attn  37.81M (37.0%)  the head's DiT trunk
    #     hist_enc / traj_enc / abs_pos_enc / traj_time_embed / denoising_time_embed
    # All of those *encode* something -- an action chunk, a history, a time, a position --
    # and those inputs live in the same spaces either way. Only the head's read-out has an
    # objective-specific meaning, so `train.py` re-zeroes `act_head`'s output Linear on
    # transfer -- exactly the state a from-scratch init leaves it in -- while keeping the
    # rest. An epsilon head is close to the negation of a velocity head, so carrying that
    # one layer over would be a worse start than none.
    #
    # `ActionExpert.head_time` is what makes the rest genuinely reusable: it feeds the
    # head a noise *fraction* under both objectives, so the pretrained time conditioning
    # is not inverted on arrival.
    #
    # Unlike `pretrained_ignore_action_layout` this does NOT need `pretrained_strict=False`
    # -- every tensor matches by name and shape, which is precisely the hazard.
    #
    #   python train.py --config finetune_libero_10_flow \
    #     --pretrained_ckpt PRETRAIN_DDIM.pt --pretrained_ignore_objective -s EXP
    #
    # (tyro renders a bool field as a bare flag, not `--flag True`.)
    #
    # Applies ONLY to `pretrained_ckpt`. Resuming with `-c` always demands an exact match.
    pretrained_ignore_objective: bool = False

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

        if self.objective not in OBJECTIVES:
            raise ValueError(
                "unknown objective '{}'; valid choices are {}"
                .format(self.objective, list(OBJECTIVES)))
        if self.flow_time_sampling not in FLOW_TIME_SAMPLING:
            raise ValueError(
                "unknown flow_time_sampling '{}'; valid choices are {}"
                .format(self.flow_time_sampling, list(FLOW_TIME_SAMPLING)))

        # Checked here rather than at injection time: a typo'd target would otherwise
        # surface after the dataloader and both backbones are already up.
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
    
    def model_kwargs(self) -> Dict:
        """The subset of this config that `models/vla.py:vla_*` takes.

        Lives here so training and evaluation cannot drift: `infer_utils/planner.py`
        rebuilds the model from the run's dumped config json, and an objective or a step
        count that only train.py knew about would silently fall back to the default at
        evaluation time. `action_norm` / `action_space` are deliberately absent -- those
        are resolved from the checkpoint itself, which outranks the json.
        """
        return dict(
            objective=self.objective,
            diffusion_timesteps=self.diffusion_timesteps,
            # 0 is the config-level spelling of "let the objective decide"; the model
            # constructor spells the same thing None.
            inference_timesteps=(self.inference_timesteps or None),
            flow_time_sampling=self.flow_time_sampling,
            flow_time_alpha=self.flow_time_alpha,
        )

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


def _flow_variant(base: TrainConfig, **overrides) -> TrainConfig:
    """A copy of `base` trained with the flow objective instead of DDIM.

    The mutable fields are re-wrapped rather than shared: `dataclasses.replace` copies
    field *values*, so the variant and its base would otherwise hold the same list
    objects, and `__post_init__`'s in-place rewrite of `dataset_classes` would run twice
    over one list.
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


# Flow-matching counterparts of the presets above. Everything except the objective is
# unchanged, so an ablation against the DDIM preset of the same name is apples-to-apples
# on data, schedule and model size.
#
# What is NOT possible: fine-tuning a released (DDIM) pretrain checkpoint with one of
# these. The two objectives share an identical state_dict layout, so the load would
# succeed silently -- `check_objective` refuses it instead. Either pretrain with
# `pretrain_flow` first, or train the fine-tune from scratch (raise max_iterations if so).
CONFIGS["pretrain_flow"] = _flow_variant(CONFIGS["pretrain"])
CONFIGS["finetune_libero_10_flow"] = _flow_variant(CONFIGS["finetune_libero_10"])
CONFIGS["finetune_real_flow"] = _flow_variant(CONFIGS["finetune_real"])
