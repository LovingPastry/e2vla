import os
import sys
import time
import tyro
import torch
import envars
import argparse
import torch.amp
from tqdm import tqdm
from typing import Dict
from datetime import datetime
from torch import Tensor, optim
from diffusers.optimization import get_scheduler

from models import vla
from models.action_norm import build_action_normalizer
from models.action_space import build_action_space
from configs import CONFIGS, TrainConfig
from models.action_expert import DEFAULT_OBJECTIVE
from train_utils.ckpt import (load_actor_weights, check_objective,
                              check_action_norm, check_action_layout, check_vlm_lora,
                              load_vlm_lora_weights)
from train_utils.lora import setup_lora, setup_vlm_lora, lora_state_dict
from train_utils.ema_impl import ExponentialMovingAverage
from train_utils.tb_logger import (TBLogger, grad_norms, param_norm,
                                   adamw_update_norm, trainable_param_table)
from models.conv_tower import CONV_BRANCH_KEYS
from data_utils.dataset_base import get_dataloader, generate_sample_weights


def conv_branch_prefixes(cfg: TrainConfig):
    """state_dict prefixes (relative to `actor`) of the modules the conv branch adds.

    Empty when no branch is configured, so the pretrained load stays byte-for-byte the
    strict one it always was.
    """
    if cfg.conv_tower == "none":
        return ()
    return tuple("context_encoder.{}.".format(k) for k in CONV_BRANCH_KEYS)


def init_train_config():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("-s", dest="save", type=str, default="", help="exp name to save")
    parser.add_argument("-c", dest="conti", type=str, default="", help="exp name to resume")

    if "-h" in sys.argv or "--help" in sys.argv:
        print("=== argparse help ===")
        parser.print_help()
        print("\n=== tyro help ===")
        tyro.extras.get_parser(TrainConfig).print_help()
        sys.exit(0)

    args, remaining_argv = parser.parse_known_args()
    save: str = args.save
    conti: str = args.conti

    cfg = CONFIGS[args.config]
    cfg = tyro.cli(cfg.__class__, default=cfg, args=remaining_argv)
    return cfg, save, conti


class AverageMeter(object):
    def __init__(self):
        self.sum = 0
        self.count = 0
    
    def reset(self):
        self.sum = 0
        self.count = 0
    
    def append(self, val):
        self.sum += val
        self.count += 1
    
    def avg(self):
        if self.count == 0:
            return 0
        else:
            return self.sum / self.count


def _every(iters: int, interval: int) -> bool:
    """`interval <= 0` means the feature is off, not "every iteration"."""
    return interval > 0 and (iters % interval == 0)


def count_trainable(m: torch.nn.Module):
    count = 0
    for p in m.parameters():
        if p.requires_grad:
            count += p.numel()
    return count


def get_data_loader_for_cfg(cfg: TrainConfig):
    if cfg.sample_multiplex > 1:
        assert cfg.dataset_weights is not None, \
            "sample_multiplex should be used together with dataset_weights"

    datasets = [D.inst() for D in cfg.dataset_classes]

    # A dataset that carries its own action space (RealBinDataset.ACTION_SPACE) decides the
    # width of history/future_actions. If that disagrees with cfg.action_space the mismatch
    # only surfaces deep inside states2action, as a reshape error on an unrelated line --
    # so check it here, where both sides are still named.
    cfg_layout = build_action_space(cfg.action_space).layout
    for ds in datasets:
        ds_space = getattr(ds, "action_space", None)
        if ds_space is not None and ds_space.layout != cfg_layout:
            raise ValueError(
                "{} 产出的是 '{}' (state_dim={})，而 TrainConfig.action_space='{}' 对应 "
                "'{}'。改 cfg.action_space 或改数据集的 ACTION_SPACE，两边必须一致。"
                .format(type(ds).__name__, ds_space.layout, ds_space.state_dim,
                        cfg.action_space, cfg_layout))

    if cfg.dataset_weights is not None:
        sample_weights = generate_sample_weights(datasets, cfg.dataset_weights)
    else:
        sample_weights = None
    
    shuffle = True if (cfg.dataset_weights is None) else None
    dataloader = get_dataloader(
        datasets=datasets,
        batch_size=cfg.bs,
        num_workers=cfg.workers,
        shuffle=shuffle,
        persistent_workers=True,
        sample_weights=sample_weights,
        sample_multiplex=cfg.sample_multiplex
    )
    return dataloader


class Trainer(object):
    LOG_DIR = "./logs/E2VLA"
    CKPT_DIR = "./checkpoints/E2VLA"

    def __init__(self):

        self.launch_time_str = datetime.now().strftime("%Y%m%d%H%M")
        self.cfg, save, conti = init_train_config()
        print("[INFO] Train config:")
        print(self.cfg)

        self.model_device = "cuda:0"
        # Built before the model: the normalizer is a constructor argument, because both
        # the dataset boundary (states2action/action2states) and the geometry inside the
        # diffusion head need the same statistics.
        self.action_space = build_action_space(self.cfg.action_space)
        print("[INFO] Action space: {} (layout '{}', action_dim={}, state_dim={})"
              .format(self.action_space.name, self.action_space.layout,
                      self.action_space.action_dim, self.action_space.state_dim))
        self.action_norm = build_action_normalizer(
            self.cfg.action_norm_stats, what="action_norm_stats",
            expect_layout=self.action_space.layout)
        if self.action_norm is None:
            print("[INFO] No action normalization (action_norm_stats is unset): the head "
                  "denoises raw camera-relative actions.")
        else:
            print("[INFO] Action normalization from {}:\n{}"
                  .format(self.cfg.action_norm_stats, self.action_norm))

        print("[INFO] Generative objective: {}".format(self.cfg.objective))
        self.model: vla.VLA = getattr(vla, "vla_" + self.cfg.model.strip())(
            action_norm=self.action_norm,
            action_space=self.action_space,
            **self.cfg.model_kwargs()).to(self.model_device)

        print("[INFO] Total {:.3f}M trainable parameters"
              .format(count_trainable(self.model) / 1e6))
        
        self.train_loader = get_data_loader_for_cfg(self.cfg)
        self.scaler = torch.amp.GradScaler(
            "cuda", 
            enabled=self.cfg.fp16
        )

        self.save = False

        # ckpt loading priority: conti > pretrained_ckpt
        #
        # These two are NOT the same operation:
        #   conti           -- resume an interrupted run. Everything (weights, optimizer
        #                      moments, LR schedule position, iteration counter) must be
        #                      restored so training continues as if never stopped.
        #   pretrained_ckpt -- start a NEW run from pretrained weights. Only the weights
        #                      carry over; see `pretrained_weights_only` below.
        if conti:
            self.save = conti  # ckpt subfolder name is same as opt.conti
            ckpt = torch.load(os.path.join(self.CKPT_DIR, conti, "ckpt_latest.pt"),
                              map_location=self.model_device,
                              weights_only=False)
            check_objective(ckpt, self.cfg.objective, what="resume checkpoint")
            check_action_layout(ckpt, self.action_space.layout,
                                what="resume checkpoint")
            # A resume must continue the same optimisation problem, action space
            # included -- no legitimate reason for these to differ.
            check_action_norm(ckpt, self.action_norm, what="resume checkpoint",
                              strict=True)
            # A resume checkpoint was written by an already-LoRA-ified model, so LoRA
            # must be injected BEFORE loading (opposite order from pretrained_ckpt).
            ckpt_rank = ckpt.get("lora_rank", 0)
            if ckpt_rank != self.cfg.lora_rank:
                raise ValueError(
                    "lora_rank mismatch: checkpoint was trained with {} but the config "
                    "says {}. Pass the same --config used for the original run."
                    .format(ckpt_rank, self.cfg.lora_rank))
            if self.cfg.lora_rank > 0:
                setup_lora(self.model.actor.context_encoder, self.cfg.lora_rank)
            # Same argument for the backbones, and the same required order: inject, then
            # load. A resume must reproduce the exact module tree it left off with.
            check_vlm_lora(ckpt, self.cfg.vlm_lora_rank, self.cfg.vlm_lora_targets,
                           what="resume checkpoint", strict=True)
            if self.cfg.vlm_lora_rank > 0:
                setup_vlm_lora(self.model.vlm, self.cfg.vlm_lora_rank,
                               self.cfg.vlm_lora_targets)
                load_vlm_lora_weights(self.model.vlm, ckpt["vlm_lora_weights"],
                                      what="resume checkpoint")
            load_actor_weights(self.model.actor, ckpt["weights"],
                               strict=True, what="resume checkpoint")
            self.current_iters = ckpt["current_iters"]
            self.last_ep = ckpt["last_ep"]
        elif self.cfg.pretrained_ckpt:
            ckpt = torch.load(self.cfg.pretrained_ckpt,
                              map_location=self.model_device,
                              weights_only=False)
            # Strict unless explicitly waived: unlike the action-norm / LoRA checks below,
            # nothing about a cross-objective load is detectable after the fact, so it
            # takes a deliberate flag rather than a warning.
            cross_objective = False
            if self.cfg.pretrained_ignore_objective:
                stored = ckpt.get("objective", DEFAULT_OBJECTIVE)
                cross_objective = stored != self.cfg.objective
                if cross_objective:
                    print("[WARN] pretrained checkpoint was trained with objective '{}', "
                          "this run uses '{}'. Proceeding on the explicit "
                          "pretrained_ignore_objective flag: 96.5% of the action expert "
                          "(ContextEncoder + the head's DiT stack + every input encoder) "
                          "encodes things whose meaning does not depend on the objective "
                          "and transfers as-is. `act_head` does not, and is re-zeroed "
                          "below."
                          .format(stored, self.cfg.objective))
            else:
                check_objective(ckpt, self.cfg.objective, what="pretrained checkpoint")
            if self.cfg.pretrained_ignore_action_layout:
                stored = ckpt.get("action_layout", "cam_rel_t3r6_openness")
                if stored != self.action_space.layout:
                    if self.cfg.pretrained_strict:
                        raise ValueError(
                            "pretrained_ignore_action_layout needs pretrained_strict=False: "
                            "the checkpoint predicts in '{}' and this run in '{}', so "
                            "hist_enc.0 / traj_enc.0 / act_head.3 genuinely differ in shape."
                            .format(stored, self.action_space.layout))
                    print("[WARN] pretrained checkpoint predicts in action layout '{}', "
                          "this run uses '{}'. Proceeding on the explicit "
                          "pretrained_ignore_action_layout flag: the shared trunk "
                          "(ContextEncoder + the head's DiT stack) transfers, and the "
                          "three layers bound to the action encoding are re-initialised. "
                          "The per-tensor report below is the authority on what loaded."
                          .format(stored, self.action_space.layout))
            else:
                check_action_layout(ckpt, self.action_space.layout,
                                    what="pretrained checkpoint")
            # Not strict: fine-tuning a released (unnormalized) pretrain checkpoint with
            # per-dataset q01/q99 statistics is a normal thing to want. It still gets a
            # loud warning, because doing it unintentionally is indistinguishable from
            # doing it on purpose right up until evaluation.
            check_action_norm(ckpt, self.action_norm, what="pretrained checkpoint",
                              strict=False)
            # Load into the PLAIN model first: released checkpoints have no LoRA keys.
            # The conv branch is new in this run, so a released checkpoint cannot carry
            # it. Naming those keys keeps `pretrained_strict` True -- everything else
            # still has to line up exactly.
            load_actor_weights(self.model.actor, ckpt["weights"],
                               strict=self.cfg.pretrained_strict,
                               what="pretrained checkpoint",
                               allow_missing_prefixes=conv_branch_prefixes(self.cfg))
            if cross_objective:
                # After the load, not before: `load_actor_weights` writes act_head like
                # any other tensor, so this has to undo it rather than pre-empt it.
                self.model.actor.dp_head.reset_output_layer()
                print("[INFO] act_head's output Linear re-zeroed, exactly as a "
                      "from-scratch init leaves it: an epsilon head is close to the "
                      "negation of a velocity head, so the trained output layer would be "
                      "a worse start than none. Everything feeding it is kept.")
            if self.cfg.lora_rank > 0:
                setup_lora(self.model.actor.context_encoder, self.cfg.lora_rank)
            # Not strict: adding VLM LoRA on top of a checkpoint trained with frozen
            # backbones is the normal way to start (the factors are a no-op at init).
            # Dropping factors the checkpoint WAS fitted with only warns -- see
            # check_vlm_lora for why that case cannot be detected any other way.
            matches = check_vlm_lora(ckpt, self.cfg.vlm_lora_rank,
                                     self.cfg.vlm_lora_targets,
                                     what="pretrained checkpoint", strict=False)
            if self.cfg.vlm_lora_rank > 0:
                setup_vlm_lora(self.model.vlm, self.cfg.vlm_lora_rank,
                               self.cfg.vlm_lora_targets)
                if matches:
                    load_vlm_lora_weights(self.model.vlm, ckpt["vlm_lora_weights"],
                                          what="pretrained checkpoint")
            self.current_iters = 0
            self.last_ep = -1
        else:
            if self.cfg.lora_rank > 0:
                # LoRA on top of random weights adapts nothing -- the frozen base it
                # decomposes around is itself untrained.
                raise ValueError(
                    "lora_rank > 0 requires --pretrained_ckpt: LoRA freezes the "
                    "ContextEncoder's base weights, which would leave 60% of the model "
                    "stuck at its random initialisation.")
            # vlm_lora_rank has no such precondition: the base it decomposes around is
            # SigLIP/DINOv2, which are pretrained no matter how this run started.
            if self.cfg.vlm_lora_rank > 0:
                setup_vlm_lora(self.model.vlm, self.cfg.vlm_lora_rank,
                               self.cfg.vlm_lora_targets)
            self.current_iters = 0
            self.last_ep = -1

        n_train = count_trainable(self.model)
        n_vlm = count_trainable(self.model.vlm)
        print("[INFO] After checkpoint setup: {:.3f}M trainable parameters "
              "({:.3f}M in the VLM backbones, {:.3f}M in the action expert)"
              .format(n_train / 1e6, n_vlm / 1e6, (n_train - n_vlm) / 1e6))

        # if save path is explicitly specified, then overwrite
        if save:
            self.save = save

        decay, no_decay, conv_decay, conv_no_decay = self.model.parameter_groups()
        groups = [
            {"params": decay, "lr": self.cfg.max_lr, "weight_decay": self.cfg.wd},
            {"params": no_decay, "lr": self.cfg.max_lr, "weight_decay": 0.0},
        ]
        # Appended only when non-empty, and that is not a tidiness choice: `-c` resume
        # calls `optimizer.load_state_dict`, which requires the same NUMBER of param
        # groups it was saved with. Two unconditional empty groups would make every
        # checkpoint written before the conv branch existed unresumable.
        #
        # `constant_with_warmup` is a LambdaLR, so it scales each group by a factor of
        # that group's own base lr -- the ratio below holds for the whole schedule.
        if conv_decay or conv_no_decay:
            conv_lr = self.cfg.max_lr * self.cfg.conv_tower_lr_scale
            groups += [
                {"params": conv_decay, "lr": conv_lr, "weight_decay": self.cfg.wd},
                {"params": conv_no_decay, "lr": conv_lr, "weight_decay": 0.0},
            ]
            n_conv = sum(p.numel() for p in conv_decay + conv_no_decay)
            print("[INFO] conv branch: {:.3f}M parameters in their own param groups at "
                  "lr {:.2e} ({}x max_lr)"
                  .format(n_conv / 1e6, conv_lr, self.cfg.conv_tower_lr_scale))
        self.optimizer = optim.AdamW(groups)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.ema = ExponentialMovingAverage(params, decay=self.cfg.ema_decay)

        # model score
        self.best_score = None
        self.larger_better = False

        if conti:
            print("[INFO] resume training from iter: {}".format(ckpt["current_iters"]))
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scaler.load_state_dict(ckpt["scaler"])
            self.best_score = ckpt["best_score"]
            if "ema" in ckpt:
                self.ema.load_state_dict(ckpt["ema"])
            else:
                print("[INFO] Key ema not found in checkpoint, skip loading ema model")
        elif self.cfg.pretrained_ckpt:
            print("[INFO] load pretrained ckpt from iter: {}".format(ckpt["current_iters"]))
            if self.cfg.pretrained_weights_only:
                # Fine-tuning starts a fresh optimisation problem: current_iters is reset
                # to 0 and the LR schedule re-warms from zero. Carrying over the
                # pretrain run's Adam moments would pair stale first/second moments with
                # a warming-up LR, which makes the first updates larger than intended --
                # exactly what you do not want on a small dataset.
                print("[INFO] pretrained_weights_only=True: optimizer/scaler/EMA state "
                      "from the pretrained ckpt is intentionally NOT restored.")
            else:
                self.optimizer.load_state_dict(ckpt["optimizer"])
                self.scaler.load_state_dict(ckpt["scaler"])
                if "ema" in ckpt:
                    self.ema.load_state_dict(ckpt["ema"])
                else:
                    print("[INFO] Key ema not found in checkpoint, skip loading ema model")

        # EMA that can never fire is worse than no EMA: `save_model` would still write an
        # "ema" entry, and `remote_service --ema` would then restore the *initial*
        # weights at eval time, silently discarding the entire run.
        if self.cfg.ema_enabled and self.cfg.ema_start >= self.cfg.max_iterations:
            raise ValueError(
                "ema_enabled=True but ema_start ({}) >= max_iterations ({}), so "
                "ema.update() would never be called and the saved EMA weights would be "
                "the untrained initial ones. Lower ema_start (e.g. just after warmup) "
                "or set ema_enabled=False."
                .format(self.cfg.ema_start, self.cfg.max_iterations)
            )

        self.scheduler = get_scheduler(
            name="constant_with_warmup",
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.num_warmup,
            last_epoch=self.last_ep,
        )
        if conti:
            self.scheduler.load_state_dict(ckpt["scheduler"])

        self._is_first_save = True

        # Opened here rather than lazily at the first log: the run's own description --
        # the resolved config, what ended up trainable, where the weights came from -- is
        # worth having in the event file even for a run that dies in its first hundred
        # iterations, which is exactly when you go looking for it.
        self.logger = TBLogger(os.path.join(self.LOG_DIR, self.save) if self.save else None)
        self.logger.text("config", "```json\n{}\n```".format(self.cfg.to_json()),
                         self.current_iters)
        self.logger.text("params", trainable_param_table(self.model), self.current_iters)
        self.logger.text("run", "\n".join([
            "* launched: `{}`".format(self.launch_time_str),
            "* action space: `{}` (layout `{}`)".format(self.action_space.name,
                                                        self.action_space.layout),
            "* objective: `{}`".format(self.cfg.objective),
            "* action_norm: `{}`".format(
                "none" if self.action_norm is None else self.cfg.action_norm_stats),
            "* resumed from: `{}`".format(conti or "-"),
            "* pretrained_ckpt: `{}`".format(self.cfg.pretrained_ckpt or "-"),
            "* datasets: {}".format(", ".join("`{}`".format(D.__name__)
                                              for D in self.cfg.dataset_classes)),
        ]), self.current_iters)

        # Wall-clock anchor for perf/it_per_sec; reset at every log interval.
        self._interval_start = time.perf_counter()

    @classmethod
    def preprocess_data(
        cls, 
        data: Dict[str, Tensor], 
        device,
    ):
        for k in data:
            if isinstance(data[k], Tensor):
                data[k] = data[k].to(device, non_blocking=True)

        return data

    def compute_metrics(self, data: Dict[str, Tensor]):
        data = self.preprocess_data(data, self.model_device)
        total_loss, metrics = self.model(
            rgbs=data["rgbs"],
            obs_norm_xys=data["obs_norm_xys"],
            obs_extrinsics=data["obs_extrinsics"],
            prompt_text=data["prompt_text"],

            ee_poses=data["ee_poses"],
            history_actions=data["history_actions"],
            future_actions=data["future_actions"], 
            valid_ee_mask=data["valid_ee_mask"], 
            inference=False,
            fp16=self.scaler.is_enabled(),
        )

        return total_loss, metrics

    def _clip_grad(self):
        """Clip the gradient if configured, returning its PRE-clip norm (None if not).

        `clip_grad_norm_` computes that norm on its way to the scale factor, so logging it
        costs nothing extra. With clipping disabled nothing is computed here -- `gnorm/*`
        in `log_metrics` reports the same quantity at the log interval, and computing it
        every iteration just to average it is not worth a second pass over 102M gradients.
        """
        if self.cfg.grad_clip <= 0:
            return None
        total = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.cfg.grad_clip)
        return float(total)

    @torch.no_grad()
    def sampled_action_metrics(self, data: Dict[str, Tensor]) -> Dict[str, float]:
        """Run the full sampler on `data` and report the action error in physical units.

        The training loss scores ONE denoising step against its target; this scores the
        trajectory that comes out of the whole loop, in metres and degrees. The two come
        apart in both directions -- a head can keep improving its epsilon regression while
        the integrated chunk drifts, and the flow/DDIM losses are not even comparable to
        each other while these numbers are -- so this is the closest thing to a rollout
        that does not need a simulator.

        On the training batch, since this repo has no validation split. It answers "can
        the sampler reproduce what it was trained on", which is the failure that comes
        first; generalization still needs `examples/libero/eval.py`.

        Measured on the live weights, not the EMA shadow. With `ema_enabled` the deployed
        policy is the shadow, so read this as a trend rather than as a prediction of what
        `remote_service --ema` will do.
        """
        was_training = self.model.training
        self.model.eval()
        pred = self.model(
            rgbs=data["rgbs"],
            obs_norm_xys=data["obs_norm_xys"],
            obs_extrinsics=data["obs_extrinsics"],
            prompt_text=data["prompt_text"],

            ee_poses=data["ee_poses"],
            history_actions=data["history_actions"],
            future_actions=data["future_actions"],
            valid_ee_mask=data["valid_ee_mask"],
            inference=True,
            fp16=self.scaler.is_enabled(),
        )  # (B, Ta, Nee, state_dim)
        if was_training:
            self.model.train()

        # Same flattening the model does internally: (B, Ta, Nee, D) -> (B', Ta, D) over
        # the valid end-effectors only. Invalid slots hold the action space's placeholder
        # fill, which would otherwise contribute a meaningless error.
        mask = data["valid_ee_mask"]
        pred_states = pred.transpose(1, 2)[mask]
        gt_states = data["future_actions"].transpose(1, 2)[mask]
        errors = self.action_space.state_error(pred_states.float(), gt_states.float())
        return {"sample/" + k: v for k, v in errors.items()}

    def log_metrics(self, averages: Dict[str, "AverageMeter"], data: Dict[str, Tensor]):
        """Everything that goes out once per `log_interval`.

        `averages` holds whatever the loop accumulated, keyed two ways: a bare name is a
        loss term and lands under `train/`, a name that already contains a "/" carries its
        own TensorBoard group (`diag/`, `optim/`, `perf/`) and is passed through. Point-in-
        time quantities -- norms, memory, the LR -- are read here rather than averaged,
        because their value at the moment of logging is the meaningful one.
        """
        if not self.logger.enabled:
            return

        step = self.current_iters
        scalars = {(k if "/" in k else "train/" + k): v.avg()
                   for k, v in averages.items()}

        # LR per optimizer group: with a conv tower the groups no longer share one value,
        # and "the LR" silently becoming group 0's is how a scaled branch goes unnoticed.
        lrs = self.scheduler.get_last_lr()
        scalars["optim/lr"] = lrs[0]
        for i, lr in enumerate(lrs[1:], start=1):
            if lr != lrs[0]:
                scalars["optim/lr_g{}".format(i)] = lr

        pnorm = param_norm(self.model)
        unorm = adamw_update_norm(self.optimizer)
        scalars["optim/param_norm"] = pnorm
        scalars["optim/update_norm"] = unorm
        if pnorm > 0:
            # The number to tune `max_lr` against: ~1e-3 is the healthy band, and it is
            # readable long before the loss says anything.
            scalars["optim/update_ratio"] = unorm / pnorm
        if self.scaler.is_enabled():
            scalars["optim/grad_scale"] = self.scaler.get_scale()

        # Gradients survive until the next zero_grad(), so they are still the ones that
        # produced the step logged above.
        scalars.update(grad_norms(self.model))

        elapsed = time.perf_counter() - self._interval_start
        if elapsed > 0:
            iters = self.cfg.log_interval
            scalars["perf/it_per_sec"] = iters / elapsed
            scalars["perf/samples_per_sec"] = iters * self.cfg.bs / elapsed
        if torch.cuda.is_available():
            scalars["perf/gpu_mem_gb"] = torch.cuda.max_memory_allocated() / (1 << 30)
            torch.cuda.reset_peak_memory_stats()

        scalars["progress/epoch"] = self.last_ep + 1
        scalars["progress/samples_seen"] = self.current_iters * self.cfg.bs
        scalars["data/valid_ee"] = float(data["valid_ee_mask"].sum(-1).float().mean())

        self.logger.scalars(scalars, step)
        self._interval_start = time.perf_counter()

    def log_inputs(self, data: Dict[str, Tensor]):
        """The batch's images and prompt, as the model sees them.

        Deliberately taken from `data` *after* `preprocess_data`, i.e. the exact tensor
        that entered the model: this is the check that catches a missing /255, a BGR
        dataset, and -- most importantly -- cameras in the wrong order, since camera 0
        defines the frame every action is expressed in. Camera 0 is the leftmost tile and
        must be the third-person view.
        """
        if not self.logger.enabled:
            return
        rgbs = data["rgbs"]              # (B, To, ncam, 3, H, W)
        images = rgbs[0, -1]             # latest frame of the first sample, (ncam, 3, H, W)
        self.logger.image_grid("data/rgb_cam0..N", images, step=self.current_iters)
        prompt = data.get("prompt_text", None)
        if isinstance(prompt, (list, tuple)) and prompt:
            self.logger.text("data/prompt", str(prompt[0]), self.current_iters)

    def save_model(self, fname: str, best_score: float, latest_score: float):
        if self.save and self._is_first_save:
            cfg_save_path = os.path.join(self.CKPT_DIR, self.save, "{}.json".format(self.launch_time_str))
            self.cfg.dump(cfg_save_path)
            self._is_first_save = False

        if self.save:
            ckpt_dir = os.path.join(self.CKPT_DIR, self.save)
            os.makedirs(ckpt_dir, exist_ok=True)
            to_save = {
                # With LoRA this state_dict carries `...to_q.lin.weight` plus `.A`/`.B`
                # instead of `...to_q.weight`, so `lora_rank` below tells every loader
                # to inject LoRA before calling load_state_dict. Deployment merges the
                # factors back into the base weights (see infer_utils/planner.py).
                "weights": self.model.actor.state_dict(),
                "lora_rank": self.cfg.lora_rank,
                # The VLM itself is never serialized -- it is rebuilt from HuggingFace /
                # torch.hub every time. So when its backbones are LoRA-adapted, these
                # factors are the ONLY record that the features the action expert was
                # trained against are not the stock ones. Losing them is not a load
                # error, it is a silently different policy; `check_vlm_lora` reads the
                # rank/targets back out to make the mismatch loud.
                "vlm_lora_rank": self.cfg.vlm_lora_rank,
                "vlm_lora_targets": list(self.cfg.vlm_lora_targets),
                # Stamp what these weights actually mean. Nothing else can tell: a head
                # trained on a different objective has the same tensor names and shapes,
                # so it would load here without a single missing key. The released
                # pretrain checkpoints predate this stamp; ours all carry it.
                "objective": self.cfg.objective,
                # same reasoning as `objective`: two action spaces with the same
                # channel count produce identical state_dict layouts
                "action_layout": self.action_space.layout,
                # The action statistics are part of what the weights mean, so they
                # travel with them. `infer_utils/planner.py` reads them back from here,
                # which keeps evaluation correct even if the JSON has moved or the
                # config file in the checkpoint dir was overwritten by a later run.
                "action_norm": (None if self.action_norm is None
                                else self.action_norm.to_dict()),
                "current_iters": self.current_iters,
                "last_ep": self.last_ep, 
                "lr": self.scheduler.get_last_lr()[0], 
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "best_score": best_score,
                "latest_score": latest_score
            }
            if self.cfg.vlm_lora_rank > 0:
                to_save["vlm_lora_weights"] = lora_state_dict(self.model.vlm)
            if self.cfg.ema_enabled:
                to_save["ema"] = self.ema.state_dict()
            torch.save(to_save, os.path.join(ckpt_dir, fname))
            # tqdm.write, not print: this fires from inside the training loop, and a bare
            # print would be overwritten by the next redraw of the progress bar.
            tqdm.write("[INFO] Save to {}".format(os.path.join(ckpt_dir, fname)))

    def fitting(self):
        averages = {}
        self.model.train()

        def record(key, value):
            if key not in averages:
                averages[key] = AverageMeter()
            averages[key].append(value)

        # One bar over the whole run rather than one per epoch: `max_iterations` is what
        # the loop actually terminates on, and an epoch bar would restart its ETA every
        # pass over the loader. `initial` picks up where a resumed checkpoint left off.
        # The bar lives on stderr (tqdm's default) while every message below goes through
        # `tqdm.write` to stdout, so `python train.py > train.log` keeps a clean log of the
        # [INFO] lines and leaves the redrawing bar on the terminal.
        pbar = tqdm(initial=self.current_iters, total=self.cfg.max_iterations,
                    desc="train", unit="it", dynamic_ncols=True, smoothing=0.1)

        while self.current_iters <= self.cfg.max_iterations:
            data_start = time.perf_counter()
            for data in self.train_loader:
                # Time spent blocked on the loader, not on the GPU. When this is a large
                # fraction of the step, the fix is `workers` / `sample_multiplex` / the
                # HDF5 layout -- nothing about the model -- and no other curve says so.
                data_wait = time.perf_counter() - data_start
                step_start = time.perf_counter()

                self.current_iters += 1
                self.optimizer.zero_grad()
                loss, metrics = self.compute_metrics(data)

                if torch.isnan(loss) or torch.isinf(loss):
                    tqdm.write("[INFO] NaN or Inf occured in loss, skip")
                    self.current_iters -= 1
                    # Counted rather than only printed: a handful of skips is normal, a
                    # rising rate is a run that is quietly training on fewer samples than
                    # its iteration counter claims.
                    record("perf/nan_skip_frac", 1.0)
                    data_start = time.perf_counter()
                    continue
                record("perf/nan_skip_frac", 0.0)

                # A scaled gradient's norm is off by the scaler's current factor, which
                # itself moves around -- so unscale before anything reads it. Free when
                # clipping already does it; on a log iteration without clipping do it
                # anyway, since `gnorm/*` is about to walk the gradients.
                is_log_iter = (self.current_iters % self.cfg.log_interval == 0)

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    if self.cfg.grad_clip > 0 or is_log_iter:
                        self.scaler.unscale_(self.optimizer)
                    grad_norm = self._clip_grad()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    grad_norm = self._clip_grad()
                    self.optimizer.step()
                self.scheduler.step()

                if grad_norm is not None:
                    record("optim/grad_norm", grad_norm)
                    if self.cfg.grad_clip > 0:
                        # How often clipping actually binds. Permanently at 1.0 means the
                        # threshold, not the LR, is setting the step size.
                        record("optim/clip_frac",
                               float(grad_norm > self.cfg.grad_clip))

                if self.current_iters >= self.cfg.ema_start and self.cfg.ema_enabled:
                    self.ema.update()

                record("perf/data_wait_ms", data_wait * 1e3)
                record("perf/step_ms", (time.perf_counter() - step_start) * 1e3)

                print_strings = []
                for key, val in metrics.items():
                    record(key, val)
                    # The `diag/*` diagnostics go to TensorBoard only; putting a dozen
                    # extra numbers on the bar would push the losses off the line.
                    if "/" not in key:
                        print_strings.append("{} = {:.3e}".format(key, averages[key].avg()))
                print_strings.append("lr = {:.3e}".format(self.scheduler.get_last_lr()[0]))

                # Set the postfix before advancing, so `update()`'s own redraw already
                # carries this iteration's numbers. Stepping the bar to `current_iters`
                # instead of by 1 keeps it honest across the NaN skip above, which rolls
                # the counter back.
                pbar.set_postfix_str(" | ".join(print_strings), refresh=False)
                pbar.update(self.current_iters - pbar.n)

                ### save ckpt and log
                if self.current_iters % self.cfg.save_latest_interval == 0:
                    avg_metrics = {k: v.avg() for k, v in averages.items()}
                    latest_score = avg_metrics["total_loss"]
                    
                    if (
                        (self.best_score is None) or
                        (self.larger_better and (latest_score > self.best_score)) or 
                        (not self.larger_better and (latest_score < self.best_score)) 
                    ):
                        self.best_score = latest_score
                        save_best = True
                    else:
                        save_best = False

                    self.save_model("ckpt_latest.pt", self.best_score, latest_score)
                    if save_best:
                        self.save_model("ckpt_best.pt", self.best_score, latest_score)
                
                if (self.current_iters % self.cfg.save_interval == 0) and (self.cfg.save_interval > 0):
                    avg_metrics = {k: v.avg() for k, v in averages.items()}
                    latest_score = avg_metrics["total_loss"]
                    self.save_model("ckpt_{:0>7d}.pt".format(self.current_iters),
                                    self.best_score, latest_score)

                # Before log_metrics, so the sampled errors land on the same step as the
                # losses that produced them. Runs the sampler, hence its own interval --
                # and hence `logger.enabled`, so a run without -s does not pay for a
                # measurement with nowhere to go.
                if (self.logger.enabled
                        and _every(self.current_iters, self.cfg.log_sample_interval)):
                    sampled = self.sampled_action_metrics(data)
                    self.logger.scalars(sampled, self.current_iters)
                    tqdm.write("[INFO] sampled action error | {}".format(" | ".join(
                        "{} = {:.4f}".format(k.split("/")[-1], v)
                        for k, v in sampled.items())))

                if _every(self.current_iters, self.cfg.log_image_interval):
                    self.log_inputs(data)

                if _every(self.current_iters, self.cfg.log_hist_interval):
                    # Before zero_grad() on the next iteration, so the gradients logged
                    # are the ones that produced the latest step.
                    self.logger.histograms(self.model, self.current_iters)

                if is_log_iter:
                    # The bar redraws in place and leaves no history, so keep the old
                    # per-iteration line -- once per `log_interval`, on the same interval
                    # boundary the averages are reset at, so what is written is exactly
                    # the window that just went to TensorBoard.
                    tqdm.write("[INFO] {}/{} | {}".format(
                        self.current_iters, self.cfg.max_iterations,
                        " | ".join(print_strings)))
                    self.log_metrics(averages, data)
                    for key in averages.keys():
                        averages[key].reset()

                if self.current_iters > self.cfg.max_iterations:
                    break

                data_start = time.perf_counter()

            self.last_ep += 1

        pbar.close()
        self.logger.close()


if __name__ == "__main__":
    trainer = Trainer()
    trainer.fitting()
