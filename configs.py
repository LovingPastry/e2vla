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
