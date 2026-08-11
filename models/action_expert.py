import torch
import torch.nn.functional as F

from torch import nn, Tensor
from diffusers import DDIMScheduler
from typing import Optional, Tuple, Dict, List

from .dit import DiT
from .action_norm import ActionNormalizer
from .action_space import ActionSpace, build_action_space, reference_cam_pose
from .context_encoder import (build_context_encoder, CONTEXT_ENCODERS,
                              DEFAULT_CONTEXT_ENCODER)
from .layers.utils import simple_mlp
from .layers.pe import SinusoidalPosEmb
from .layers.rot_transforms import matrix_to_rotation_6d, rotation_6d_to_matrix


# The generative objectives the action head can be trained with. This tuple is the single
# source of truth: `configs.TrainConfig` validates against it and `train_utils/ckpt.py`
# stamps the chosen value into every checkpoint.
#
#   "ddim" -- DDIM epsilon prediction, `diffusion_timesteps` train steps, ~20 at inference
#   "flow" -- rectified-flow / optimal-transport matching, ~10 Euler steps at inference
#
# Both drive the *same* network: `DiffusionHead` maps (time, x_t, context) to a vector of
# action_dim. Only three things around it differ -- how x_t is built from the clean chunk,
# what the output is supervised against, and how the sampler integrates it. Which is
# exactly why the checkpoint stamp has to exist: the two share every tensor name and every
# tensor shape, so a flow checkpoint loads into a DDIM model with zero missing keys, prints
# a clean layout match, and then feeds a velocity field to a noise-prediction sampler.
OBJECTIVES = ("ddim", "flow")

# What a checkpoint carrying no "objective" key is. The released pretrain checkpoints
# predate the stamp and are all DDIM.
DEFAULT_OBJECTIVE = "ddim"

# How the flow time t is drawn during training. t = 0 is pure noise, t = 1 is data.
FLOW_TIME_SAMPLING = ("uniform", "logitnormal", "beta")

# How many noise-level buckets `diffusion_diagnostics` splits the regression error over.
DIAG_NOISE_BINS = 4


@torch.no_grad()
def diffusion_diagnostics(
    pred: Tensor,
    target: Tensor,
    noise_level: Tensor,
    actions: Optional[Tensor] = None,
    num_bins: int = DIAG_NOISE_BINS,
) -> Dict[str, float]:
    """Per-step scalars the loss curve cannot show. Keys are prefixed `diag/`.

    The loss is one number averaged over everything, and the three ways this head goes
    wrong all leave it looking healthy:

    * **collapse to the mean.** A head that ignores its time conditioning still minimises
      the loss -- it just predicts one averaged field, and samples blur toward the mean
      action. `diag/pred_std` decaying away from `diag/target_std` is that happening, and
      `diag/cos_sim` (scale-free, so unaffected by the loss weights) stalling near 0 is
      the same thing seen from the other side.
    * **one end of the path untrained.** `diag/loss_noise_b{i}` splits the *unweighted* L1
      by how corrupted the input was: bin 0 is nearly clean, bin `num_bins - 1` nearly
      pure noise. A high-noise bin that never comes down means every sampler step
      inherits its error; that is the case `flow_time_sampling="beta"` exists for.
    * **saturated normalization.** `diag/act_clip_frac` is the fraction of target channels
      outside [-1, 1]. With `action_norm_stats` set those are exactly the actions falling
      outside the q01/q99 range the statistics were fitted on -- a few percent is normal,
      a large fraction means the stats came from a different dataset or a different
      config, which nothing else in the pipeline reports.

    Args:
        pred / target: (B, Ta, action_dim), whatever the objective regresses.
        noise_level: (B,) in [0, 1], 0 = clean, 1 = pure noise. Under DDIM that is
            `timesteps / num_train_timesteps`; under flow it is `1 - t`.
        actions: (B, Ta, action_dim), the clean (normalized) chunk. Optional.

    Everything is stacked into one tensor before the single `.tolist()`, so this costs one
    GPU sync rather than one per scalar.
    """
    pred = pred[..., :target.shape[-1]].float()
    target = target.float()

    err = (pred - target).abs().mean(dim=(1, 2))  # (B,) unweighted, per sample
    cos = F.cosine_similarity(pred.flatten(1), target.flatten(1), dim=1)  # (B,)

    bins = (noise_level.float() * num_bins).long().clamp(0, num_bins - 1)
    bin_sum = torch.zeros(num_bins, device=err.device, dtype=err.dtype)
    bin_cnt = torch.zeros(num_bins, device=err.device, dtype=err.dtype)
    bin_sum.index_add_(0, bins, err)
    bin_cnt.index_add_(0, bins, torch.ones_like(err))

    scalars = [err.mean(), cos.mean(), pred.std(), target.std()]
    if actions is not None:
        act = actions.float()
        scalars.append(act.abs().amax())
        scalars.append((act.abs() > 1.0).float().mean())
    num_head = len(scalars)

    values = torch.cat([torch.stack(scalars), bin_sum, bin_cnt]).tolist()

    out = {
        "diag/abs_err": values[0],
        "diag/cos_sim": values[1],
        "diag/pred_std": values[2],
        "diag/target_std": values[3],
    }
    if actions is not None:
        out["diag/act_absmax"] = values[4]
        out["diag/act_clip_frac"] = values[5]
    for i in range(num_bins):
        count = values[num_head + num_bins + i]
        if count > 0:
            # Empty bins are omitted rather than logged as 0: a bucket that no sample
            # landed in is not an error of zero.
            out["diag/loss_noise_b{}".format(i)] = values[num_head + i] / count
    return out


class DiffusionHead(nn.Module):
    """Conditional vector field over a chunk of camera-frame end-effector actions.

    - Input: the encoded observation context and the corrupted action chunk at time `t`
    - Output: one action_dim vector per chunk position

    The module is deliberately objective-agnostic -- what that output *means* is decided
    by `ActionExpert.objective`, not here:
      * "ddim" -- it is the noise mixed into the chunk, and the scheduler derives step t-1
      * "flow" -- it is the velocity dx_t/dt, and the sampler takes an Euler step along it
    Nothing in this class changes between the two, which is why a checkpoint cannot be
    told apart by its tensors and needs the objective stamp (see `OBJECTIVES` above).

    Two different "times" run through this module; keep them apart:
      * `timestep`     -- the denoising / flow time, a global condition driving adaLN/FiLM.
                          Always arrives already scaled to [0, diffusion_timesteps) --
                          see `ActionExpert.head_time` for why flow's t in [0,1] must not
                          be passed raw.
      * chunk position -- where an action sits inside the horizon, an additive
                          sinusoidal embedding (`traj_time_embed`)

    NOTE on the history encoder: `hist_enc` is built for `action_dim - 1` inputs and is
    fed `history[..., :action_dim-1]`, i.e. the t3r6 pose delta only. Gripper openness
    (the last channel) is deliberately withheld from history so the model cannot simply
    copy the previous gripper command; it has to read openness off the wrist camera.

    NOTE on normalization: when `action_norm` is set, everything this module sees --
    `history`, `noisy_actions`, and the prediction -- lives in the *normalized*
    action space. The one exception is `pos_rel2abs`, which is geometry: it interprets
    its input as a metric SE(3) delta, so the t3r6 channels are unnormalized right
    before that call. Feeding it normalized values would silently produce a wrong
    absolute position and the loss would barely notice.
    """

    def __init__(self, hdim: int, num_heads: int, action_space: ActionSpace, num_layers: int,
                 action_norm: Optional[ActionNormalizer] = None):
        super().__init__()
        self.action_space = action_space
        action_dim = action_space.action_dim
        self.action_dim = action_dim
        self.num_layers = num_layers
        # not a submodule assignment by accident: ActionNormalizer's buffers are
        # non-persistent, so sharing the instance with ActionExpert adds no state_dict keys
        self.action_norm = action_norm
        # module attribute names are load-bearing: they are the checkpoint state_dict keys
        self.hist_enc = simple_mlp([action_dim-1, hdim, hdim], ln=True)
        self.traj_enc = simple_mlp([action_dim, hdim, hdim], ln=True)
        # Only built for action spaces whose actions are SE(3) deltas. In joint space
        # recovering the absolute ee position would need forward kinematics (a URDF this
        # repo does not carry), and an unbuilt module keeps it out of the state_dict
        # instead of leaving dead, gradient-less weights in every checkpoint.
        self.abs_pos_enc = (simple_mlp([3, hdim, hdim], ln=True)
                            if action_space.has_pose_geometry else None)
        self.traj_time_embed = SinusoidalPosEmb(hdim)
        self.denoising_time_embed = nn.Sequential(
            SinusoidalPosEmb(hdim),
            simple_mlp([hdim, hdim, hdim], ln=True)
        )

        ### traj self attn + traj-context cross attn
        self.traj_context_attn = DiT(hdim, num_heads, num_layers, use_adaln=True)

        ### final mlp
        self.act_head = simple_mlp([hdim, hdim, action_dim], ln=True)
        self.reset_parameters()

    def reset_parameters(self):
        # Zero both weight AND bias. A zeroed weight with a random bias still injects a
        # constant offset, which defeats the point of starting these branches at zero:
        # `abs_pos_enc` must not perturb the features at init, and `act_head` must
        # predict exactly zero at init (zero noise under DDIM, a zero velocity field
        # under flow -- in both cases the sampler starts as the identity on its input).
        if self.abs_pos_enc is not None:
            nn.init.zeros_(self.abs_pos_enc[-1].weight)
            nn.init.zeros_(self.abs_pos_enc[-1].bias)
        self.reset_output_layer()

    def reset_output_layer(self):
        """Re-zero `act_head` -- the only layer whose meaning depends on the objective.

        Everything else in this module *encodes* something: an action chunk, a history, a
        time, a position. Those inputs live in the same spaces under either objective, so
        the weights that read them keep their meaning. The output layer does not. An
        epsilon head predicts the noise; a flow head predicts `actions - noise`, which at
        the high-noise end of the path is very nearly the *negation* of it. Carrying the
        trained layer across therefore starts the new run with a systematically
        sign-flipped output -- strictly worse than the zero it would have had from
        scratch, and worse in a way that looks like slow convergence rather than a bug.

        Called on its own by `train.py` when `pretrained_ignore_objective` transfers a
        trunk across the objective boundary.
        """
        nn.init.zeros_(self.act_head[-1].weight)
        nn.init.zeros_(self.act_head[-1].bias)

    def pos_rel2abs(self, cur_wcT: Tensor, cur_weT: Tensor, t3r6: Tensor):
        """Relative action encoding -> absolute ee position in the camera frame.

        Actions are stored as a delta from the *current* ee pose, expressed in the
        camera's orientation (see `space_ee2cam`). That representation is translation
        invariant, which is what we want to predict, but it carries no information
        about where in the workspace the arm actually is. This recovers the absolute
        position so it can be fed back as an additional positional encoding.

        Mirrors `space_cam2ee`, but stops in the camera frame instead of going all the
        way back to world: ^{cam}T_{ee} = (^{world}T_{cam})^-1 @ ^{world}T_{ee} @ delta

        Args:
            cur_wcT (Tensor): (B, 4, 4), ^{world} T _{cam}
            cur_weT (Tensor): (B, 4, 4), ^{world} T _{ee}
            t3r6 (Tensor): (B, T, 9), 3 translation + 6D rotation, camera-relative

        Returns:
            traj_cet (Tensor), traj ee pos in camera frame, shape (B, T, 3)
        """
        ecT = torch.inverse(cur_weT) @ cur_wcT  # (B, 4, 4)
        ecR = ecT[:, :3, :3]  # (B, 3, 3)
        
        e1e2R = ecR[:, None] @ rotation_6d_to_matrix(t3r6[..., 3:]) @ ecR[:, None].transpose(-1, -2)
        e1e2t = (ecR[:, None] @ t3r6[..., :3].unsqueeze(-1)).squeeze(-1)
        
        e1e2T = e1e2t.new_zeros(*e1e2t.shape[:-1], 4, 4)
        e1e2T[..., :3, :3] = e1e2R
        e1e2T[..., :3, 3] = e1e2t
        e1e2T[..., 3, 3] = 1

        traj_ceT = (torch.inverse(cur_wcT) @ cur_weT)[:, None] @ e1e2T
        traj_cet = traj_ceT[..., :3, 3]  # (B, T, 3)
        return traj_cet

    def forward(
        self,
        timestep: Tensor,
        noisy_actions: Tensor,
        cur_wcT: Tensor,
        cur_weT: Tensor,
        history: Tensor,
        conds: List[Tensor],
        cond_masks: Optional[List[Optional[Tensor]]],
        fp16: bool
    ):
        """One network evaluation.

        History and the noisy action chunk are concatenated into a single sequence so
        self-attention can relate the two; only the action segment is read out at the
        end. History acts as clean, always-available conditioning.

        Args:
            timestep: (B,), the denoising / flow time, already scaled to
                [0, diffusion_timesteps) (NOT the position within the chunk)
            noisy_actions: (B, Ta, action_dim), the corrupted action chunk at `timestep`
            cur_wcT: (B, 4, 4), ^{world} T _{cam}
            cur_weT: (B, 4, 4), ^{world} T _{ee}
            history: (B, nhist, action_dim), past actions in the same encoding
            conds: [(B, Lc, hdim)], observation context from ContextEncoder
            cond_masks: [(B, Lc)] or [None]
            fp16 (bool): use bfloat16 autocast for the attention stack

        Returns:
            pred: (B, Ta, action_dim), noise or velocity depending on the objective
        """
        time_embed = self.denoising_time_embed(timestep)  # (B, hdim)
        film = time_embed

        batch_size, history_horizon, _ = history.shape
        batch_size, action_horizon, _ = noisy_actions.shape
        # gripper openness is intentionally dropped from history (see class docstring)
        history_feats = self.hist_enc(history[:, :, :self.action_dim-1])  # (B, nhist, hdim)
        action_feats = self.traj_enc(noisy_actions[:, :, :self.action_dim])  # (B, Ta, hdim)

        # additive sinusoidal PE over position within [history ; chunk]
        seq_feats = torch.cat([history_feats, action_feats], dim=1)  # (B, nhist+Ta, hdim)
        seq_pos_pe = self.traj_time_embed(
            torch.arange(history_horizon + action_horizon).to(action_feats))
        seq_feats = seq_feats + seq_pos_pe[None].expand(batch_size, -1, -1)

        # absolute ee position in the camera frame as an extra positional encoding.
        # no_grad wraps only the geometry: the actions are the variable being denoised,
        # so this is treated as a coordinate lookup, not a differentiable path. The
        # abs_pos_enc call itself stays outside so its weights still get gradients.
        if self.abs_pos_enc is not None:
            with torch.no_grad():
                seq_t3r6 = torch.cat(
                    [history[..., :9], noisy_actions[..., :9]], dim=1)  # (B, nhist+Ta, 9)
                if self.action_norm is not None:
                    # back to metres / a real rotation before doing SE(3) algebra on it
                    seq_t3r6 = self.action_norm.unnormalize(seq_t3r6)
                abs_pos = self.pos_rel2abs(cur_wcT, cur_weT, seq_t3r6)
            seq_feats = seq_feats + self.abs_pos_enc(abs_pos)

        with torch.autocast(
            time_embed.device.type,
            torch.bfloat16 if fp16 else torch.float32
        ):
            seq_feats = self.traj_context_attn(
                x=seq_feats,
                x_pe=None,
                x_mask=None,
                conds=conds,
                cond_masks=cond_masks,
                films=[film]*len(conds)
            )

        # drop the history segment; only the chunk is supervised
        action_feats = seq_feats[:, history_horizon:history_horizon+action_horizon]
        pred = self.act_head(action_feats)
        return pred


class ActionExpert(nn.Module):
    def __init__(
        self, 
        hdim: int, 
        num_heads: int, 
        num_context_layers: int,
        num_diffusion_layers: int, 
        diffusion_timesteps: int = 100,
        inference_timesteps: Optional[int] = None,
        action_norm: Optional[ActionNormalizer] = None,
        action_space: Optional[str | ActionSpace] = None,
        objective: str = DEFAULT_OBJECTIVE,
        flow_time_sampling: str = "uniform",
        flow_time_alpha: float = 1.5,
        conv_tower: Optional[str] = None,
        context_encoder: str = DEFAULT_CONTEXT_ENCODER,
    ):
        super().__init__()
        if objective not in OBJECTIVES:
            raise ValueError("unknown objective '{}'; valid choices are {}"
                             .format(objective, list(OBJECTIVES)))
        if context_encoder not in CONTEXT_ENCODERS:
            raise ValueError("unknown context_encoder '{}'; valid choices are {}"
                             .format(context_encoder, list(CONTEXT_ENCODERS)))
        if flow_time_sampling not in FLOW_TIME_SAMPLING:
            raise ValueError("unknown flow_time_sampling '{}'; valid choices are {}"
                             .format(flow_time_sampling, list(FLOW_TIME_SAMPLING)))
        self.objective = objective
        self.flow_time_sampling = flow_time_sampling
        self.flow_time_alpha = float(flow_time_alpha)
        # None == the EE-pose space, i.e. the historical behaviour.
        self.action_space = build_action_space(action_space)
        # q01/q99 normalization of the action space. None reproduces the pre-normalization
        # behaviour exactly. The same instance is handed to the head so both sides of the
        # train/inference boundary use one set of statistics.
        if action_norm is not None and action_norm.action_dim != self.action_dim:
            raise ValueError(
                "action stats cover {} channels but the model's action_dim is {}. The "
                "stats file was computed for a different action layout."
                .format(action_norm.action_dim, self.action_dim))
        self.action_norm = action_norm

        self.context_encoder_name = context_encoder
        self.context_encoder = build_context_encoder(
            context_encoder, hdim=hdim, num_heads=num_heads,
            num_layers=num_context_layers, conv_tower=conv_tower)
        # The one combination that loads, runs and then means nothing: a camera-relative
        # action space needs `^{world}T_{cam}` for every sample, and a vision-action
        # encoder is built precisely so no calibration reaches the model. Substituting
        # identity silently would redefine what the checkpoint's actions mean without
        # changing a single tensor -- exactly the failure mode `action_layout` exists to
        # prevent -- so it is refused here instead.
        if (self.action_space.uses_camera_pose
                and not self.context_encoder.uses_camera_params):
            raise ValueError(
                "context_encoder='{}' reads no camera parameters, but action_space '{}' "
                "(layout '{}') expresses every action in camera 0's frame, which needs "
                "the extrinsics. Use action_space='ee_base' (the same encoding rebased "
                "onto the robot's own frame) or a joint space."
                .format(context_encoder, self.action_space.name,
                        self.action_space.layout))
        self.dp_head = DiffusionHead(hdim, num_heads, self.action_space,
                                     num_layers=num_diffusion_layers,
                                     action_norm=action_norm)

        # Only built for DDIM. Flow matching has no noise schedule to speak of -- the
        # forward process is a straight line and the sampler is plain Euler -- so leaving
        # the scheduler as None keeps `objective` the single thing that decides behaviour
        # rather than having a live-but-unused scheduler lying around to be picked up by
        # accident.
        if objective == "ddim":
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=diffusion_timesteps,
                beta_schedule="squaredcos_cap_v2",
                prediction_type="epsilon",
                clip_sample=False
            )
        else:
            self.noise_scheduler = None

        # Under "flow" this is no longer a count of anything; it survives as the numeric
        # range the head's sinusoidal time embedding is defined over. See `head_time`.
        self.diffusion_timesteps = diffusion_timesteps
        if inference_timesteps is None:
            # Flow needs far fewer function evaluations than DDIM for the same quality --
            # that, and not the loss, is the practical reason to switch objectives here.
            inference_timesteps = (max(diffusion_timesteps//5, 10) if objective == "ddim"
                                   else 10)
        self.inference_timesteps = inference_timesteps
        self.inference_scheduler = self.noise_scheduler

    @property
    def action_dim(self):
        """Model-side action width, set by the configured `ActionSpace`.

        EE-pose: 10 (3 translation + 6D rotation + 1 openness, camera-relative).
        Joint:   nq + 1. Note this differs from `state_dim`, the width the dataset hands
        over (17 for EE-pose), which the two are no longer guaranteed to share.
        """
        return self.action_space.action_dim

    def head_time(self, t: Tensor):
        """Flow time t in [0, 1] -> what the head's time embedding is actually fed.

        NOTE THE FLIP: this returns `(1 - t) * diffusion_timesteps`, so t and the value
        the head sees run in *opposite* directions. Two separate reasons, and both are
        needed.

        Scale. `dp_head.denoising_time_embed` starts with a
        `SinusoidalPosEmb(temperature=1e4)` designed around integer DDIM steps spread over
        [0, diffusion_timesteps). Handing it a raw t in [0, 1] would compress every
        training sample into the first 1% of that range, where all but the very highest
        frequency bands are flat -- the head would be nearly time-blind and would settle
        for one averaged velocity field instead of a t-dependent one. That failure is
        quiet: the loss still drops (the average field is a real minimiser), the samples
        just blur toward the mean action.

        Direction. What these embeddings conventionally index is *noise level*, and the
        two objectives number it the other way round: a DDIM timestep of 0 is clean data
        and `diffusion_timesteps` is pure noise, whereas flow's t = 0 is pure noise and
        t = 1 is data. Feeding `1 - t` makes the argument mean "noise fraction" under both
        -- the same thing pi0 feeds, since its t is already the noise fraction. Without
        the flip a DDIM-pretrained head would start out with its notion of "how corrupted
        is my input" exactly inverted, which is the difference between a useful init and
        an actively misleading one (see `TrainConfig.pretrained_ignore_objective`).

        Both properties are bijections on [0, 1], so a from-scratch flow run is
        indifferent to either choice; only transfer and readability care.
        """
        return (1.0 - t) * self.diffusion_timesteps

    def sample_time(self, batch_size: int, device) -> Tensor:
        """Draw the training-time flow times, (B,) in [0, 1]. t=0 is noise, t=1 is data.

        Which end of the path gets the most supervision is the one real hyperparameter of
        flow matching, hence three choices:

        * "uniform"     -- the rectified-flow baseline. Every t is equally important.
        * "logitnormal" -- sigmoid(N(0,1)), from SD3. Concentrates on the middle of the
                           path, where the field actually changes; the two ends are close
                           to trivial (near t=0 the answer is nearly `-noise`, near t=1
                           nearly the residual).
        * "beta"        -- 1 - Beta(alpha, 1), pi0's choice restated in this module's t
                           convention (pi0 runs t the other way round). Mass sits near
                           t=0, the high-noise end: errors made there are integrated by
                           every remaining Euler step, so they cost the most.

        Uniform is the default because it is the objective as published and has no knob;
        the other two are worth trying if sampling with few steps looks under-converged.
        """
        if self.flow_time_sampling == "uniform":
            return torch.rand(batch_size, device=device)
        if self.flow_time_sampling == "logitnormal":
            return torch.sigmoid(torch.randn(batch_size, device=device))
        if self.flow_time_sampling == "beta":
            beta = torch.distributions.Beta(self.flow_time_alpha, 1.0)
            return 1.0 - beta.sample((batch_size,)).to(device)
        raise ValueError("unknown flow_time_sampling '{}'".format(self.flow_time_sampling))

    def sample_actions(
        self,
        actions_shape: Tuple[int, int, int],
        fixed_inputs: Dict[str, Tensor],
        initial_noise: Optional[Tensor] = None
    ):
        """Pure noise -> an action chunk, by whichever sampler the objective calls for.

        `fixed_inputs` is everything that does not change across sampler steps (the
        observation context, the current camera/ee poses, the action history); it is
        computed once by `forward` and splatted into the head each step.

        Args:
            actions_shape: (B, Ta, action_dim)
            fixed_inputs: keyword arguments forwarded to `dp_head`
            initial_noise: (B, Ta, action_dim), sampled if None

        Returns:
            actions: (B, Ta, action_dim)
        """
        if initial_noise is None:
            batch_size, action_horizon, _ = actions_shape
            device = next(iter(fixed_inputs.values())).device
            initial_noise = torch.randn(batch_size, action_horizon, self.action_dim,
                                        device=device)

        if self.objective == "flow":
            return self.flow_integrate(fixed_inputs, initial_noise)
        return self.iterative_denoise(fixed_inputs, initial_noise)

    def iterative_denoise(self, fixed_inputs: Dict[str, Tensor], initial_noise: Tensor):
        """Full DDIM reverse loop. See `sample_actions`."""
        self.inference_scheduler.set_timesteps(self.inference_timesteps)
        actions = initial_noise
        for t in self.inference_scheduler.timesteps:
            pred_noise = self.dp_head(
                t * torch.ones(actions.shape[0], device=actions.device),
                actions,
                **fixed_inputs
            )
            actions = self.inference_scheduler.step(
                pred_noise[..., :self.action_dim], t, actions[..., :self.action_dim]
            ).prev_sample
        return actions

    def flow_integrate(self, fixed_inputs: Dict[str, Tensor], initial_noise: Tensor):
        """Euler integration of the learned velocity field, t: 0 (noise) -> 1 (data).

        Uniform steps and no scheduler, deliberately. On the OT path the true trajectory
        between a noise sample and its paired action chunk is a straight line travelled at
        constant speed, so the discretisation itself contributes no error -- everything
        that remains is the network's own deviation from the marginal field, which no
        step-size schedule can anticipate. `inference_timesteps` is therefore exactly the
        number of network evaluations, and 10 is usually enough where DDIM wanted 20.

        See `sample_actions` for the arguments.
        """
        num_steps = self.inference_timesteps
        dt = 1.0 / num_steps
        actions = initial_noise
        for i in range(num_steps):
            # left endpoint of the step: the field is evaluated where we currently are.
            # t counts UP from 0 (noise) to 1 (data); head_time flips it, so the value the
            # head sees counts down from diffusion_timesteps -- see `head_time`.
            t = torch.full((actions.shape[0],), i * dt,
                           device=actions.device, dtype=actions.dtype)
            velocity = self.dp_head(self.head_time(t), actions, **fixed_inputs)
            actions = actions + dt * velocity[..., :self.action_dim]
        return actions

    def forward(
        self, 
        vl_obs: Dict[str, Tensor],
        vl_feature: Dict[str, Tensor], 
        ee_poses: Tensor, 
        history_actions: Tensor, 
        future_actions: Tensor, 
        valid_ee_mask: Tensor, 
        inference: bool, 
        fp16: bool,
    ):
        """
        Args:
            vl_obs (Dict[str, Tensor]): the language and camera-parameter entries are
                None under a vision-action context encoder -- see `models/vlm.py`
                - rgb: (B, To, ncam, 3, H, W)
                - norm_xy: (B, To, ncam, 2, H, W) or None
                - text: List (length=B) of prompt, or None
                - extrinsics: (B, To, ncam, 4, 4) or None, ^{world}_{camera} T

            vl_feature (Dict[str, Tensor]):
                - norm_xy_ds: (B, Ncam, Lv, 2) or None
                - vision_embeds: List (length=num_layer) of (B, Ncam, Lv, C)
                - lang_embeds: List (length=num_layer) of (B, La, C), or None
                - lang_mask: (B, La) or None
                - extrinsics: (B, Ncam, 4, 4) or None

            ee_poses: (B, Nee, 4, 4), ^{world}_{ee} T
            history_actions: (B, nhist, Nee, 4*4+1), in world frame,
                * 4x4 is the flattened transformation matrix, 
                * 1 is gripper openness, range [0 (close), 1 (open)]
            future_actions: (B, Ta, Nee, 4*4+1), ground truth future actions, in world frame
                * 4x4 is the flattened transformation matrix, 
                * 1 is gripper openness, range [0 (close), 1 (open)]
                * Note: if `inference` is True, we only derive prediction actions shape from future_actions
            valid_ee_mask: (B, Nee), only compute loss on these end-effectors
            inference: if True, returns the predicted trajectory, otherwise returns loss and metrics for logging
            fp16: if True, use bfloat16
        
        Returns
        -------
        (if inference is True)
            pred_future_actions (Tensor): (B, Ta, Nee, 4*4+1)
                * 4x4 is the flattened transformation matrix, 
                * 1 is gripper openness, range [0 (close), 1 (open)]
        (else)
            loss (Tensor): scalar tensor
            metrics (Dict[str, Tensor]): metrics for logging
        """
        # The frame the whole action representation is expressed in. Under "ee_cam" that
        # is camera 0 at the latest timestep -- the same camera `VLContextEncoder` builds
        # its PRoPE relative to. Under a camera-free action space it is the identity, and
        # `vl_obs["extrinsics"]` is not read at all (it is None in that case).
        current_cam_pose = reference_cam_pose(
            self.action_space, vl_obs["extrinsics"],
            batch_size=vl_obs["rgb"].shape[0], like=ee_poses)  # (B, 4, 4)

        # patch features as current observation context in diffusion
        cond, cond_mask = self.context_encoder(
            vl_obs=vl_obs,
            vl_feature=vl_feature,
            fp16=fp16,
        )

        # Flatten (B, Nee) -> B' by keeping only the valid end-effectors. Each valid ee
        # becomes an independent sample sharing its batch element's observation context.
        # `batch_index` maps flat position -> original batch index; boolean masking with
        # valid_ee_mask walks (B, Nee) in row-major order, so the two orders agree.
        valid_ee_per_batch = valid_ee_mask.sum(dim=-1)  # (B,)
        batch_index = torch.cat([torch.empty(n, dtype=torch.long).fill_(b)
                                 for b, n in enumerate(valid_ee_per_batch.tolist())]
                                ).to(valid_ee_mask.device)
        flat_batch_size = len(batch_index)  # B'

        # (B, nhist, Nee, state_dim) -> (B, Nee, nhist, state_dim) so the ee axis is maskable
        history_action_cam = self.action_space.states2action(
            current_cam_pose[batch_index],
            ee_poses[valid_ee_mask],
            history_actions.transpose(1, 2)[valid_ee_mask],
            self.action_norm
        )  # (B', nhist, action_dim)

        batch_size, action_horizon, num_ee, _ = future_actions.shape
        if not inference:
            future_action_cam = self.action_space.states2action(
                current_cam_pose[batch_index],
                ee_poses[valid_ee_mask],
                future_actions.transpose(1, 2)[valid_ee_mask],
                self.action_norm
            )  # (B', Ta, action_dim)

        # everything the denoiser needs that is constant across denoising steps
        fixed_inputs = dict(
            history=history_action_cam,  # history in camera 0, shape (B', nhist, action_dim)
            conds=[cond[batch_index]],
            cond_masks=[cond_mask[batch_index] if cond_mask is not None else cond_mask],
            cur_wcT=current_cam_pose[batch_index],  # (B', 4, 4)
            cur_weT=ee_poses[valid_ee_mask],        # (B', 4, 4)
            fp16=fp16
        )

        ###################### Inference ######################
        if inference:
            pred_actions = self.sample_actions(
                actions_shape=(flat_batch_size, action_horizon, self.action_dim),
                fixed_inputs=fixed_inputs
            )  # (B', Ta, action_dim)
            pred_future_actions = self.action_space.action2states(
                current_cam_pose[batch_index],  # (B', 4, 4)
                ee_poses[valid_ee_mask],        # (B', 4, 4)
                pred_actions,  # (B', Ta, action_dim)
                self.action_norm
            )  # (B', Ta, state_dim)

            # scatter B' back to (B, Nee); invalid slots get the action space's neutral
            # fill (identity pose for EE, zeros for joints)
            pred_future_actions_full = pred_future_actions.new_zeros(
                batch_size, num_ee, action_horizon, self.action_space.state_dim)
            pred_future_actions_full = self.action_space.init_invalid_states(
                pred_future_actions_full)
            pred_future_actions_full[valid_ee_mask] = pred_future_actions
            # (B, Ta, Nee, state_dim)
            return pred_future_actions_full.transpose(1, 2).contiguous()

        ###################### Training ######################
        # sample noise
        noise = torch.randn(flat_batch_size, action_horizon, self.action_dim,
                            device=future_actions.device)

        if self.objective == "flow":
            # Rectified flow / optimal-transport path. The forward process is a straight
            # line between a noise sample and the clean chunk,
            #     x_t = (1 - t) * noise + t * actions,   t in [0, 1]
            # so its velocity is constant along the path and the regression target is
            # just the endpoint difference. Same convention as
            # pvrobo/src/agent/flow_policy.py -- t = 1 is data, which is the opposite of
            # pi0's; if you port code between them, flip t and negate the velocity.
            t = self.sample_time(flat_batch_size, noise.device)  # (B',)
            t_expand = t[:, None, None]
            noisy_actions = (1 - t_expand) * noise + t_expand * future_action_cam
            # head_time, not t: the head's sinusoidal embedding is defined over
            # [0, diffusion_timesteps), see `head_time`
            pred = self.dp_head(self.head_time(t), noisy_actions, **fixed_inputs)
            target = future_action_cam - noise
            # what `diffusion_diagnostics` buckets by: 0 = clean, 1 = pure noise. Flow's
            # t runs the other way (t = 1 is data), hence the flip -- same convention as
            # `head_time`, so the bins mean the same thing under both objectives.
            noise_level = 1.0 - t
        else:
            # sample a random timestep
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                size=(flat_batch_size,),
                device=noise.device
            )

            # forward diffusion, then a single denoising step (not the full loop)
            noisy_actions = self.noise_scheduler.add_noise(
                future_action_cam, noise,
                timesteps
            )

            pred = self.dp_head(timesteps, noisy_actions, **fixed_inputs)
            target = get_target(future_action_cam, noise, timesteps, self.noise_scheduler)
            noise_level = timesteps.float() / self.noise_scheduler.config.num_train_timesteps

        # # drop too aggresive actions
        # debug_gt_ee_pose = future_actions.transpose(1, 2)[valid_ee_mask][..., :16].reshape(flat_batch_size, -1, 4, 4)
        # debug_gt_ee_pos = debug_gt_ee_pose[..., :3, 3]  # (B', Ta, 3)
        # delta_norm = (debug_gt_ee_pos[:, 1:] - debug_gt_ee_pos[:, :-1]).norm(dim=-1)
        # debug_mask = delta_norm < 0.2  # (B', Ta-1)
        # debug_mask[..., 1:-1] = debug_mask[..., 1:-1] & debug_mask[..., :-2] & debug_mask[..., 2:]
        # debug_mask = torch.cat([debug_mask[:, 0:1], debug_mask], dim=-1)  # (B', Ta)
        # # filter
        # pred = pred[debug_mask]
        # target = target[debug_mask]

        # Loss is split per action channel only so the terms can be weighted and logged
        # separately; the split itself is a property of the encoding, hence it lives on
        # the action space.
        #
        # Under "ddim" the weights are pure loss shaping: every slice of an epsilon target
        # is part of the same standard normal. Under "flow" the target is
        # `actions - noise`, which inherits the action space's own per-channel scale, so
        # the same weights now also act as a (crude) dimensional correction. Both are
        # fine, but it does mean the two objectives' loss values are NOT comparable --
        # judge a flow run by its rollout, not by putting its curve next to a DDIM one.
        total_loss, metrics = self.action_space.loss(pred, target)
        # Logging-only. Keys carry a "diag/" prefix, which is how `train.py` tells them
        # apart from the loss terms: the console line stays the loss terms only, while
        # TensorBoard gets both.
        metrics.update(diffusion_diagnostics(pred, target, noise_level,
                                             actions=future_action_cam))
        return total_loss, metrics


def space_ee2cam(cur_wcT: Tensor, cur_weT: Tensor, fut_weT: Tensor):
    """World-frame future ee poses -> the camera-relative action the model predicts.

    This is the encoding side of the action representation, and the exact inverse of
    `space_cam2ee`. Two things happen:

    1. Absolute -> relative: the future pose becomes a delta from the current ee pose,
       ^{e1}T_{e2} = (^{w}T_{e1})^-1 @ ^{w}T_{e2}. The policy predicts motion, not
       absolute placement, so it transfers across workspace positions.
    2. ee frame -> camera orientation: the delta is conjugated by ^{cam}R_{ee} so it is
       expressed in the camera's axes. This is what ties the action to what the model
       can actually see; get the direction of this rotation wrong and the loss still
       converges while the robot moves along the wrong axes.

    Only the rotation ceR is used for the conjugation (not the full transform): a delta
    is a relative quantity, so it must be rotated, never translated.

    Args:
        cur_wcT (Tensor): (B, 4, 4), ^{world} T _{cam}
        cur_weT (Tensor): (B, 4, 4), ^{world} T _{ee}
        fut_weT (Tensor): (B, T, 4, 4), future ee pose in world frame

    Returns:
        t3r6 (Tensor): 3 translation + 6D rotation, camera-relative, shape (B, T, 9)
    """
    e1e2T = torch.inverse(cur_weT[:, None]) @ fut_weT  # (B, T, 4, 4)
    e1e2R = e1e2T[:, :, :3, :3]  # (B, T, 3, 3)
    e1e2t = e1e2T[:, :, :3, 3]  # (B, T, 3)

    ceT = torch.inverse(cur_wcT) @ cur_weT  # (B, 4, 4)
    ceR = ceT[:, :3, :3]  # (B, 3, 3)
    
    r = matrix_to_rotation_6d(ceR[:, None] @ e1e2R @ ceR[:, None].transpose(-1, -2))
    t = (ceR[:, None] @ e1e2t.unsqueeze(-1)).squeeze(-1)
    t3r6 = torch.cat([t, r], dim=-1)
    return t3r6


def space_cam2ee(cur_wcT: Tensor, cur_weT: Tensor, t3r6: Tensor):
    """Camera-relative action -> world-frame future ee poses. Inverse of `space_ee2cam`.

    Note `ecT` here is (^{world}T_{ee})^-1 @ ^{world}T_{cam}, i.e. the *opposite*
    direction from `space_ee2cam`'s `ceT`, which is what makes this the inverse rather
    than a repeat of the same rotation.

    Args:
        cur_wcT (Tensor): (B, 4, 4), ^{world} T _{cam}
        cur_weT (Tensor): (B, 4, 4), ^{world} T _{ee}
        t3r6 (Tensor): (B, T, 9), 3 translation + 6D rotation, camera-relative

    Returns:
        fut_weT (Tensor), future ee pose in world frame, shape (B, T, 4, 4)
    """
    ecT = torch.inverse(cur_weT) @ cur_wcT  # (B, 4, 4)
    ecR = ecT[:, :3, :3]  # (B, 3, 3)
    
    e1e2R = ecR[:, None] @ rotation_6d_to_matrix(t3r6[..., 3:]) @ ecR[:, None].transpose(-1, -2)
    e1e2t = (ecR[:, None] @ t3r6[..., :3].unsqueeze(-1)).squeeze(-1)
    
    e1e2T = e1e2t.new_zeros(*e1e2t.shape[:-1], 4, 4)
    e1e2T[..., :3, :3] = e1e2R
    e1e2T[..., :3, 3] = e1e2t
    e1e2T[..., 3, 3] = 1

    fut_weT = cur_weT[:, None] @ e1e2T
    return fut_weT


def states2action(cur_wcT: Tensor, cur_weT: Tensor, ee_states: Tensor,
                  action_norm: Optional[ActionNormalizer] = None):
    """Dataset ee states -> model action space. The dataset/model boundary.

    Three transforms compose here, in this order:
      1. `space_ee2cam` on the pose -- absolute world SE(3) to a camera-frame delta
      2. gripper openness from the dataset's [0, 1] to [-1, 1]
      3. `action_norm`, the optional per-channel q01/q99 affine

    Steps 1-2 are fixed structure; step 3 is data-dependent and is what the JSON stats
    file supplies. Order matters: the statistics in the file are defined over the output
    of steps 1-2, because that is where the model's action space actually lives.

    Args:
        cur_wcT (Tensor): (B, 4, 4), ^{world} T _{cam}
        cur_weT (Tensor): (B, 4, 4), ^{world} T _{ee}
        ee_states (Tensor): (B, T, 16 or 17), flattened 4x4 pose [+ openness in [0,1]]
        action_norm: q01/q99 normalizer, or None to skip step 3

    Returns:
        action (Tensor): (B, T, 9 or 10), t3r6 [+ openness in [-1,1]]
    """
    B, Ta, C = ee_states.shape
    # 数据集给的是关节角（nq+1 宽）而 action_space 仍是 ee_cam 时，下面的 view 会抛一句
    # 看不出病因的 "shape ... is invalid for input of size ..."。这里先拦住，对齐
    # AbsJoint.states2action 的断言。
    assert C in (16, 17), \
        ("ee_cam 动作空间期望 state_dim=16 或 17（展平 4x4 位姿 [+ 夹爪]），实得 {}。"
         "若数据集是关节空间（nq+1），请把 TrainConfig.action_space 设成 'joint{}'"
         .format(C, C - 1))
    weT = ee_states[:, :, :16].view(B, Ta, 4, 4)
    t3r6 = space_ee2cam(cur_wcT, cur_weT, weT)

    if C == 16:
        action = t3r6
    else:
        openness = (ee_states[:, :, -1:] - 0.5) * 2  # rescale gripper openness
        action = torch.cat([t3r6, openness], dim=-1)

    if action_norm is not None:
        action = action_norm.normalize(action)
    return action


def action2states(cur_wcT: Tensor, cur_weT: Tensor, action: Tensor,
                  action_norm: Optional[ActionNormalizer] = None):
    """Model action space -> ee states the robot can execute. Inverse of `states2action`.

    Args:
        cur_wcT (Tensor): (B, 4, 4), ^{world} T _{cam}
        cur_weT (Tensor): (B, 4, 4), ^{world} T _{ee}
        action (Tensor): (B, T, 9 or 10), t3r6 [+ openness in [-1,1]]
        action_norm: the same normalizer `states2action` was given, or None

    Returns:
        ee_states (Tensor): (B, T, 16 or 17), flattened 4x4 pose [+ openness in [0,1]]
    """
    if action_norm is not None:
        action = action_norm.unnormalize(action)

    B, Ta, C = action.shape
    t3r6 = action[:, :, :9]
    weT = space_cam2ee(cur_wcT, cur_weT, t3r6).view(B, Ta, 16)

    if C == 9:
        return weT
    else:
        openness = action[:, :, -1:] / 2 + 0.5  # back to the dataset's [0,1]
        return torch.cat([weT, openness], dim=-1)


def get_target(actions: Tensor, noise: Tensor, timesteps: Tensor, scheduler: DDIMScheduler):
    """Supervision target matching the scheduler's parameterisation. DDIM only.

    The flow objective has no scheduler and no `prediction_type`; its target is written
    inline in `ActionExpert.forward` because it is one subtraction.

    Args:
        actions (Tensor): (B, Ta, action_dim), the clean action chunk
        noise (Tensor): (B, Ta, action_dim), the noise mixed into it
        timesteps (Tensor): (B,), the sampled diffusion steps
        scheduler: supplies `prediction_type`
    """
    pred_type = scheduler.config.prediction_type
    if pred_type == "epsilon":
        target = noise
    if pred_type == "sample":
        target = actions
    if pred_type == "v_prediction":
        target = scheduler.get_velocity(actions, noise, timesteps)
    return target


def count_parameters():
    """One line per context encoder, at the `base` width the presets actually use."""
    for name in CONTEXT_ENCODERS:
        model = ActionExpert(
            hdim=768,
            num_heads=12,
            num_context_layers=8,
            num_diffusion_layers=4,
            diffusion_timesteps=100,
            # every VA encoder refuses a camera-relative action space, see __init__
            action_space="ee_cam" if name == "vl" else "ee_base",
            context_encoder=name,
        )

        num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_ctx = sum(p.numel() for p in model.context_encoder.parameters()
                      if p.requires_grad)
        print("[INFO] context_encoder={:>12}: {:.3f}M trainable ({:.3f}M in the context "
              "encoder, {:.3f}M in the head)"
              .format(name, num_param / 1e6, num_ctx / 1e6, (num_param - num_ctx) / 1e6))


if __name__ == "__main__":
    count_parameters()

