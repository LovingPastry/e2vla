import os
import cv2
import copy
import glob
import torch
import threading
import numpy as np
from torch import Tensor
from typing import Union

from models import vla
from .ensemble import TrajEnsembler
from data_utils.dataset_base import DataSampler, DataConfig, gen_norm_xy_map, rbd
from train_utils.lora import setup_lora, setup_vlm_lora, merge_lora_linear
from train_utils.ckpt import (check_objective, check_action_layout,
                             load_vlm_lora_weights, read_vlm_lora_spec)
from models.action_norm import build_action_normalizer
from models.action_space import build_action_space
from train_utils.ema_impl import ExponentialMovingAverage
from data_utils.datasets import DATA_CONFIGS
from .draw_traj import visualize_traj
from configs import TrainConfig


def parse_config(ckpt_dir: str):
    config_files = glob.glob(os.path.join(ckpt_dir, "*.json"))
    config_files.sort()

    assert len(config_files), "No config files found in {}".format(ckpt_dir)
    config_file = config_files[-1]
    print("[INFO] Use config file {}".format(config_file))

    cfg = TrainConfig.load(config_file)
    data_config = cfg.dataset_classes[0].config

    data_config.shuffle_cameras = False  # overwrite
    print("[INFO] model = {}".format(cfg.model))
    print("[INFO] data config = {}".format(data_config))

    # The whole config, not a tuple of the fields evaluation happens to need today: every
    # new model-shaping option (objective, sampler steps, ...) has to reach the model here
    # or evaluation silently rebuilds a *different* network from the one that was trained,
    # and a tuple return makes forgetting one the path of least resistance.
    return cfg, data_config


def load_model(path, device, use_ema: bool = False):
    """Build the policy and restore trained weights, handling LoRA checkpoints.

    Order matters and is not interchangeable:
      1. inject LoRA, so the module tree matches the checkpoint's key names
      2. load_state_dict
      3. copy EMA (its shadow tensors are the LoRA factors, so this must precede merge)
      4. merge LoRA into the base Linears -- inference then runs at zero LoRA overhead
         and the live model is a plain ActionExpert again

    Steps 1-3 apply to VLM-side LoRA too, and step 3 is why they have to be interleaved
    rather than done backbone-first: the EMA shadow is one flat list over
    `model.parameters()`, so every LoRA factor anywhere in the model must already exist,
    and none of them may have been merged away yet, at the moment it is copied back.
    Step 4 does NOT apply to the backbones -- see the note next to it.
    """
    cfg, data_config = parse_config(os.path.dirname(path))

    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Refuse a checkpoint trained on a different generative objective before building
    # anything: its weights would load with zero missing keys and then sample garbage,
    # and no later step could notice. Released checkpoints predate the stamp and are
    # read as DDIM; anything this repo trained carries it explicitly.
    #
    # The two records being compared come from different places -- the objective the
    # weights were stamped with, and the objective the run's config json says to build --
    # so this also catches a config json overwritten by a later run in the same directory.
    check_objective(ckpt, cfg.objective, what="checkpoint")
    print("[INFO] objective = {} ({} sampler steps)"
          .format(cfg.objective, cfg.inference_timesteps or "default"))

    # Same guard one level down: an EE-pose and a joint checkpoint can have identical
    # state_dict layouts, and picking the wrong one here means the robot executes joint
    # angles as if they were metres. Resolved from the run's config json and verified
    # against the checkpoint's own stamp.
    action_space = build_action_space(cfg.action_space)
    check_action_layout(ckpt, action_space.layout, what="checkpoint")
    print("[INFO] action space = {} (layout '{}')"
          .format(action_space.name, action_space.layout))

    # Action normalization, resolved the same way lora_rank is: the checkpoint's own
    # record wins over the config json, which may have been overwritten by a later run
    # in the same directory. Falling back to the json's path is only a convenience for
    # checkpoints written before the stats were stamped in.
    #
    # Getting this wrong is undetectable at runtime -- the model would emit normalized
    # units and the robot would execute them as metres -- so it is resolved once, here,
    # and never re-derived downstream.
    if "action_norm" in ckpt:
        action_norm = build_action_normalizer(
            ckpt["action_norm"], what="checkpoint action_norm",
            expect_layout=action_space.layout)
        source = "checkpoint"
    else:
        action_norm = build_action_normalizer(cfg.action_norm_stats,
                                              what=str(cfg.action_norm_stats),
                                              expect_layout=action_space.layout)
        source = "config json ({})".format(cfg.action_norm_stats)

    if action_norm is None:
        print("[INFO] No action normalization (from {})".format(source))
    else:
        print("[INFO] Action normalization from {}:\n{}".format(source, action_norm))

    model: vla.VLA = getattr(vla, "vla_{}".format(cfg.model))(
        action_norm=action_norm, action_space=action_space,
        **cfg.model_kwargs()).to(device)

    # the checkpoint's own record wins over the config json, which may have been
    # overwritten by a later run in the same directory
    lora_rank = ckpt.get("lora_rank", cfg.lora_rank)
    if lora_rank > 0:
        print("[INFO] Checkpoint was trained with LoRA rank {}".format(lora_rank))
        setup_lora(model.actor.context_encoder, lora_rank)

    # Same resolution rule, and it matters more here: the VLM's weights are not in the
    # checkpoint at all, so skipping this step produces a model that loads perfectly and
    # feeds the action expert stock features it was never trained on.
    vlm_lora_rank, vlm_lora_targets = read_vlm_lora_spec(ckpt)
    if vlm_lora_rank == 0 and "vlm_lora_rank" not in ckpt:
        vlm_lora_rank, vlm_lora_targets = cfg.vlm_lora_rank, cfg.vlm_lora_targets
    if vlm_lora_rank > 0:
        print("[INFO] Checkpoint was trained with VLM LoRA rank {} on {}"
              .format(vlm_lora_rank, vlm_lora_targets))
        setup_vlm_lora(model.vlm, vlm_lora_rank, vlm_lora_targets)
        if "vlm_lora_weights" not in ckpt:
            raise RuntimeError(
                "checkpoint declares VLM LoRA (rank {}) but carries no "
                "'vlm_lora_weights'. The adapted backbones cannot be reconstructed -- "
                "the factors are the only copy of them.".format(vlm_lora_rank))
        load_vlm_lora_weights(model.vlm, ckpt["vlm_lora_weights"], what="checkpoint")

    model.actor.load_state_dict(ckpt["weights"])
    print("[INFO] Load weights from iter: {}".format(ckpt["current_iters"]))

    if use_ema:
        param = [p for p in model.parameters() if p.requires_grad]
        ema = ExponentialMovingAverage(param, 1.0)
        ema.load_state_dict(ckpt["ema"])
        ema.to(device)
        ema.copy_to(param)
        print("[INFO] EMA weights loaded")

    if lora_rank > 0:
        merge_lora_linear(model.actor.context_encoder, inplace=True)
        print("[INFO] LoRA merged into the base weights")

    # NOTE: the VLM's factors are deliberately NOT merged, unlike the action expert's.
    # Step 4 above assumes merging is free, and for the expert it is -- its weights are
    # fp32. The backbones are bf16, where `round_bf16(W + AB)` costs ~3.3e-3 relative
    # error on W no matter how small AB is. Measured on a 768-wide bf16 Linear: a LoRA
    # whose effect on the output is 2.3e-3 gets 2.7e-3 of merge error on top, i.e. the
    # adaptation is entirely inside the rounding noise, and at a tenth of that scale only
    # 36% of AB survives the round at all. Keeping the factors live costs one rank-r
    # matmul per attention projection -- nothing against a ViT forward -- and makes
    # evaluation compute exactly what training computed.
    if vlm_lora_rank > 0:
        print("[INFO] VLM LoRA kept unmerged (bf16 backbones; see the note in load_model)")

    # Nothing here is trained again, and saying so puts the backbones back on their
    # `no_grad` fast path -- `maybe_no_grad` keys on exactly this. That matters now that
    # the VLM's factors stay live: a caller who forgets `torch.inference_mode()` would
    # otherwise tape the whole ViT forward on every observation.
    model.requires_grad_(False)
    model.eval()
    return model, data_config


class TrajPlanner(object):
    def __init__(
        self, 
        ckpt_path: str, 
        device: str = "cuda:0", 
        ensemble: int = -1,
        use_ema: bool = False
    ):
        self.model, self.config = load_model(ckpt_path, device, use_ema)

        self.ensemble = int(ensemble)
        self.ensembler_lock = threading.Lock()
        self._build_ensemblers(self.ensemble)

        self.obs_frames = []
        self.obs_lock = threading.Lock()

        self.device = device
        self.last_obs_data = None
        # set via set_prompt(); declared here so a missing call fails with a clear message
        self.prompt_text = None

    def _build_ensemblers(self, ensemble: int):
        """(Re)create the three ensemblers. The averaging window is the deque `maxlen`,
        which is fixed at construction time -- changing `ensemble` therefore requires
        rebuilding, not just resetting."""
        self.pos_ensembler = TrajEnsembler(int(ensemble))
        self.rot_ensembler = TrajEnsembler(int(ensemble))
        self.gripper_ensembler = TrajEnsembler(int(ensemble))

    def reset(self):
        """Clear all per-episode state. Call between episodes: the ensemblers key on
        timestamps, which restart at zero for a new episode."""
        with self.ensembler_lock:
            self.pos_ensembler.reset()
            self.rot_ensembler.reset()
            self.gripper_ensembler.reset()
        with self.obs_lock:
            self.obs_frames.clear()
        return self
    
    def set_config(self, config: Union[str, dict, DataConfig]):
        if isinstance(config, str):
            config = DATA_CONFIGS[config]
        elif isinstance(config, dict):
            config = DataConfig(**config)
        elif isinstance(config, DataConfig):
            pass
        else:
            raise TypeError("Unsupported type of config: {}".format(type(config)))
        
        config: DataConfig = copy.deepcopy(config)
        config.shuffle_cameras = False  # do not shuffle cameras when inference
        self.config = config
    
    def set_prompt(self, prompt_text: str):
        """
        Args:
            prompt_text (str):
        """
        self.prompt_text = prompt_text
        return self
    
    def add_obs_frame(self, obs_frame: dict):
        """
        Args:
            obs_frame (dict) should contains necessary keys listed as followings.

            - CAM_NAME_0: 
                - model: pinhole
                - camera:
                    - width: int
                    - height: int
                    - K: np.ndarray of shape 9 (3x3), flattened
                - data:
                    - color: np.ndarray, shape=(H, W, C)
                    - seg: None | np.ndarray of shape (H, W) | isaacsim seg output
                    - wcT: np.ndarray of shape (4, 4), ^{world}_{cam} T
                    - timestep: float, current timestamp used for sync
            
            - CAM_NAME_1: similar as CAM_NAME_0
            - ee_pose: np.ndarray of shape (4, 4), ^{world}_{ee} T
            - gripper: float, value from [0 (close), 1 (open)]
            - timestamp: float
        """
        max_frames = max(
            self.config.num_history_cameras * self.config.sample_camera_gaps,
            self.config.num_history_states * self.config.sample_state_gaps
        )
        
        def max_time(a: float, b: float):
            if a is None: return b
            elif b is None: return a
            else: return max(a, b)
        
        if (self.config.record_dt is None) and (self.config.sample_dt is None):
            with self.obs_lock:
                self.obs_frames.append(obs_frame)
                while len(self.obs_frames) > max_frames:
                    self.obs_frames.pop(0)
        else:
            time_interval = max_frames * max_time(self.config.record_dt, self.config.sample_dt)
            latest_time = obs_frame["timestamp"]
            earliest_time_thersh = latest_time - time_interval
            with self.obs_lock:
                self.obs_frames.append(obs_frame)
                pop_counts = 0
                for frame in self.obs_frames[1:]:
                    if frame["timestamp"] < earliest_time_thersh:
                        pop_counts += 1
                    else:
                        break                
                if pop_counts > 0:
                    self.obs_frames = self.obs_frames[pop_counts:]
            
        return self
    
    def _make_data_for_infer(self, obs_frames: list):
        """Turn the raw observation ring buffer into a batch-of-1 model input.

        Runs the exact same `DataSampler.sample_framedict` used at training time, so
        temporal alignment, image resizing and the intrinsics adjustment that goes with
        it stay identical between train and eval. `latest=True` anchors the window on
        the most recent frame instead of a random one.

        Args:
            obs_frames (list[dict]): observation ring buffer, see `add_obs_frame`

        Returns:
            obs_data (dict): tensors already on `self.device`, each with a leading
                batch dim of 1
        """
        (
            rgbs, obs_cam_poses, obs_ee_poses,
            history_actions, future_actions, current_time, K, valid_ee_mask
        ) = DataSampler.sample_framedict(
            obs_traj=obs_frames,
            ee_indices=self.config.ee_indices,
            camera_names=self.config.camera_names,
            num_history_cameras=self.config.num_history_cameras,
            num_history_states=self.config.num_history_states,
            num_future_states=self.config.num_future_states,
            latest=True,
            sample_camera_gaps=self.config.sample_camera_gaps,
            sample_state_gaps=self.config.sample_state_gaps,
            sample_dt=self.config.sample_dt,
            record_dt=self.config.record_dt,
            output_image_hw=self.config.output_image_hw,
        )

        T, ncam, C, H, W = rgbs.shape
        norm_xys = gen_norm_xy_map(H, W, K).astype(np.float32)
        norm_xys = norm_xys[None].repeat(T, axis=0)  # (T, ncam, 2, H, W)

        obs_data = {
            "K": K,                                 # (ncam, 3, 3)
            "rgbs": rgbs,                           # (T, ncam, 3, H, W)
            "prompt_text": [self.prompt_text],      # [str]
            "obs_norm_xys": norm_xys,               # (To, ncam, 2, H, W)
            "obs_extrinsics": obs_cam_poses,        # (To, ncam, 4, 4)
            "ee_poses": obs_ee_poses[-1],           # (nee, 4, 4)
            "history_actions": history_actions,     # (nhist, nee, 17)
            "future_actions": future_actions,       # (Ta, nee, 17)
            "timestamps": np.array(current_time),   # scalar
            "valid_ee_mask": valid_ee_mask,         # (nee,)
        }
        
        for k in obs_data:
            if isinstance(obs_data[k], np.ndarray):
                obs_data[k] = (torch.from_numpy(obs_data[k])
                                    .to(self.device)
                                    .unsqueeze(0))
        return obs_data
    
    def _run_inference(self, obs_data):
        for k in obs_data:
            if isinstance(obs_data[k], Tensor):
                obs_data[k] = obs_data[k].to(self.device, non_blocking=True)

        with torch.inference_mode():
            actions: Tensor = self.model(
                rgbs=obs_data["rgbs"],
                obs_norm_xys=obs_data["obs_norm_xys"],
                obs_extrinsics=obs_data["obs_extrinsics"],
                prompt_text=obs_data["prompt_text"],

                ee_poses=obs_data["ee_poses"],
                history_actions=obs_data["history_actions"],
                future_actions=obs_data["future_actions"], 
                valid_ee_mask=obs_data["valid_ee_mask"],
                inference=True,
                fp16=True,
            )  # (B, Ta, nee, 17)
        return actions
    
    def _make_empty_action(self, batch_size, action_horizon, num_ee):
        """Identity-pose / zero-gripper placeholder for end-effectors the policy did
        not predict, so downstream code never sees an all-zero (singular) 4x4."""
        actions = np.zeros((batch_size, action_horizon, num_ee, 16 + 1))
        actions[..., :16] = np.eye(4).ravel()
        return actions

    @staticmethod
    def _count_end_effectors(ee_pose: np.ndarray):
        """Number of end-effectors in a frame's `ee_pose` entry.

        `add_obs_frame` accepts both layouts (matching DataSampler.sample_framedict):
        (4, 4) for the legacy single-arm case and (Nee, 4, 4) for multi-arm. Reading
        `shape[0]` blindly returns 4 for the single-arm layout.
        """
        return ee_pose.shape[0] if ee_pose.ndim == 3 else 1

    def _scatter_to_original_order(
        self,
        nee_total: int,
        ee_indices: tuple,
        action_selected: np.ndarray
    ):
        """Undo the `ee_indices` selection done at sampling time.

        The policy only predicts the end-effectors listed in `config.ee_indices`;
        this scatters them back to their original slots so callers can index by the
        robot's own end-effector id.

        Args:
            nee_total: number of end-effectors the robot actually has
            ee_indices: original slot of each predicted end-effector, in order
            action_selected: (B, Ta, len(ee_indices), 17)

        Returns:
            action_full: (B, Ta, nee_total, 17), unpredicted slots left as identity
        """
        batch_size, action_horizon, _, _ = action_selected.shape
        action_full = self._make_empty_action(batch_size, action_horizon, nee_total)

        for i, ee_ind in enumerate(ee_indices):
            action_full[:, :, ee_ind] = action_selected[:, :, i]
        return action_full

    def get_action(
        self, 
        draw_traj: bool = False,
        compress_traj_img: bool = False
    ):
        """
        Returns
        -------
            future_ee_poses (np.ndarray): shape (Ta, 4, 4), ^{world} _{ee} T
            future_grippers (np.ndarray): shape (Ta,), range [0 (close), 1 (open)]
            future_time (np.ndarray): shape (Ta,)
            traj_img (np.ndarray | None): shape (H, Ncam*W, C) if not compressed else (nbytes,)
        """
        assert self.prompt_text is not None, \
            "No prompt set; call set_prompt() before get_action()."

        with self.obs_lock:
            obs_frames = self.obs_frames.copy()  # shallow copy

        if len(obs_frames) == 0:
            return None
        
        obs_data = self._make_data_for_infer(obs_frames)
        actions = self._run_inference(obs_data)
        
        if draw_traj:
            traj_img = visualize_traj(
                data=rbd(obs_data),
                future_ee_states=[actions[0]],
                colors=[(0, 0, 255)]
            )
            if traj_img.dtype == np.float32:
                traj_img = (traj_img * 255.).clip(0, 255).astype(np.uint8)
            if compress_traj_img:
                traj_img = cv2.imencode(".jpg", traj_img)[1]
        else:
            traj_img = None
        
        self.last_obs_data = obs_data
        actions = actions.detach().cpu().numpy()  # (B, Ta, nee_sel, 17)
        actions = self._scatter_to_original_order(
            nee_total=self._count_end_effectors(obs_frames[-1]["ee_pose"]),
            ee_indices=self.config.ee_indices,
            action_selected=actions
        )
        batch_size, action_horizon, num_ee, _ = actions.shape

        ee_poses = np.reshape(actions[..., :16], (batch_size, action_horizon, num_ee, 4, 4))
        grippers = actions[..., -1]  # (B, Ta, nee)

        # the policy predicts states at multiples of the *sampling* period, which is
        # sample_state_gaps env steps -- not one env step
        latest_time = obs_data["timestamps"][0].item()
        action_dt = self.config.sample_dt * self.config.sample_state_gaps
        future_time = (1 + np.arange(action_horizon)) * action_dt + latest_time
        future_ee_poses = ee_poses[0]  # (Ta, nee, 4, 4)
        future_grippers = grippers[0]  # (Ta, nee)

        return future_ee_poses, future_grippers, future_time, traj_img
    
    def set_ensemble_nums(self, n: int):
        """Set how many overlapping chunks are averaged. Rebuilds the ensemblers so the
        new window actually takes effect (`reset()` alone keeps the old deque maxlen)."""
        with self.ensembler_lock:
            self.ensemble = int(n)
            self._build_ensemblers(self.ensemble)

    def ensemble_traj(
        self, 
        future_ee_poses: np.ndarray,
        future_grippers: np.ndarray,
        future_time: np.ndarray
    ):
        if self.ensemble != 0:
            with self.ensembler_lock:
                future_ee_poses[..., :3, 3] = self.pos_ensembler.update(
                    future_ee_poses[..., :3, 3], future_time, on_SO3=False
                )
                # future_ee_poses[..., :3, :3] = self.rot_ensembler.update(
                #     future_ee_poses[..., :3, :3], future_time, on_SO3=True
                # )
                future_grippers = self.gripper_ensembler.update(
                    future_grippers, future_time, on_SO3=False
                )
        
        return future_ee_poses, future_grippers

