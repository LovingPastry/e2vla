import os
import sys
import tyro
import torch
import envars
import argparse
import torch.amp
from typing import Dict
from datetime import datetime
from torch import Tensor, optim
from diffusers.optimization import get_scheduler
from torch.utils.tensorboard import SummaryWriter

from models import vla
from models.action_norm import build_action_normalizer
from models.action_space import build_action_space
from configs import CONFIGS, TrainConfig
from train_utils.ckpt import (load_actor_weights, check_objective,
                              check_action_norm, check_action_layout, OBJECTIVE)
from train_utils.lora import setup_lora
from train_utils.ema_impl import ExponentialMovingAverage
from data_utils.dataset_base import get_dataloader, generate_sample_weights


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

        self.model: vla.VLA = getattr(vla, "vla_" + self.cfg.model.strip())(
            action_norm=self.action_norm,
            action_space=self.action_space).to(self.model_device)

        print("[INFO] Total {:.3f}M trainable parameters"
              .format(count_trainable(self.model) / 1e6))
        
        self.train_loader = get_data_loader_for_cfg(self.cfg)
        self.scaler = torch.amp.GradScaler(
            "cuda", 
            enabled=self.cfg.fp16
        )

        self.save = False
        self.writer = None
        
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
            check_objective(ckpt, what="resume checkpoint")
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
            load_actor_weights(self.model.actor, ckpt["weights"],
                               strict=True, what="resume checkpoint")
            self.current_iters = ckpt["current_iters"]
            self.last_ep = ckpt["last_ep"]
        elif self.cfg.pretrained_ckpt:
            ckpt = torch.load(self.cfg.pretrained_ckpt,
                              map_location=self.model_device,
                              weights_only=False)
            check_objective(ckpt, what="pretrained checkpoint")
            check_action_layout(ckpt, self.action_space.layout,
                                what="pretrained checkpoint")
            # Not strict: fine-tuning a released (unnormalized) pretrain checkpoint with
            # per-dataset q01/q99 statistics is a normal thing to want. It still gets a
            # loud warning, because doing it unintentionally is indistinguishable from
            # doing it on purpose right up until evaluation.
            check_action_norm(ckpt, self.action_norm, what="pretrained checkpoint",
                              strict=False)
            # Load into the PLAIN model first: released checkpoints have no LoRA keys.
            load_actor_weights(self.model.actor, ckpt["weights"],
                               strict=self.cfg.pretrained_strict,
                               what="pretrained checkpoint")
            if self.cfg.lora_rank > 0:
                setup_lora(self.model.actor.context_encoder, self.cfg.lora_rank)
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
            self.current_iters = 0
            self.last_ep = -1

        n_train = count_trainable(self.model)
        print("[INFO] After checkpoint setup: {:.3f}M trainable parameters"
              .format(n_train / 1e6))

        # if save path is explicitly specified, then overwrite
        if save:
            self.save = save

        decay, no_decay = self.model.parameter_groups()
        self.optimizer = optim.AdamW([
            {"params": decay, "lr": self.cfg.max_lr, "weight_decay": self.cfg.wd},
            {"params": no_decay, "lr": self.cfg.max_lr, "weight_decay": 0.0}
        ])
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

    def log_metrics(self, metrics: dict):
        if self.save:
            if self.writer is None:
                log_dir = os.path.join(self.LOG_DIR, self.save)
                os.makedirs(log_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir)
            self.writer.add_scalar(
                "lr", self.scheduler.get_last_lr()[0], self.current_iters)
            for key, val in metrics.items():
                self.writer.add_scalar(key, val, self.current_iters)

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
                # Stamp what these weights actually mean. Nothing else can tell: a head
                # trained on a different objective has the same tensor names and shapes,
                # so it would load here without a single missing key. The released
                # pretrain checkpoints predate this stamp; ours all carry it.
                "objective": OBJECTIVE,
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
            if self.cfg.ema_enabled:
                to_save["ema"] = self.ema.state_dict()
            torch.save(to_save, os.path.join(ckpt_dir, fname))
            print("[INFO] Save to {}".format(os.path.join(ckpt_dir, fname)))

    def fitting(self):
        averages = {}
        self.model.train()
        while self.current_iters <= self.cfg.max_iterations:
            for data in self.train_loader:
                self.current_iters += 1
                self.optimizer.zero_grad()
                loss, metrics = self.compute_metrics(data)

                if torch.isnan(loss) or torch.isinf(loss):
                    print("[INFO] NaN or Inf occured in loss, skip")
                    self.current_iters -= 1
                    continue

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    if self.cfg.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()
                self.scheduler.step()

                if self.current_iters >= self.cfg.ema_start and self.cfg.ema_enabled:
                    self.ema.update()

                print_strings = []
                for key, val in metrics.items():
                    if key not in averages:
                        averages[key] = AverageMeter()
                    averages[key].append(val)
                    print_strings.append("{} = {:.3e}".format(key, averages[key].avg()))

                print("[INFO] {}/{} | {} | lr = {:.3e}".format(
                    self.current_iters, self.cfg.max_iterations, " | ".join(print_strings),
                    self.scheduler.get_last_lr()[0]))

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
                
                if self.current_iters % self.cfg.log_interval == 0:
                    avg_metrics = {"train/"+k: v.avg() for k, v in averages.items()}
                    self.log_metrics(avg_metrics)
                    for key in averages.keys():
                        averages[key].reset()
                
                if self.current_iters > self.cfg.max_iterations:
                    break
            
            self.last_ep += 1


if __name__ == "__main__":
    trainer = Trainer()
    trainer.fitting()
