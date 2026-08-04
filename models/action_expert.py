import torch
import torch.nn.functional as F

from einops import rearrange
from torch import nn, Tensor
from diffusers import DDIMScheduler
from typing import Optional, Tuple, Dict, List

from .dit import DiT
from .qformer import QFormerITM
from .action_norm import ActionNormalizer
from .action_space import ActionSpace, build_action_space
from .layers.utils import simple_mlp
from .layers.utils import concat_mask
from .layers.pe import SinusoidalPosEmb, se3_inverse
from .layers.attn_dn import FFWSelfAttentionLayers, init_xncoder
from .layers.rot_transforms import matrix_to_rotation_6d, rotation_6d_to_matrix


class ContextEncoder(nn.Module):
    """Frozen VLM features -> a short, fixed-length context for the diffusion head.

    Pipeline: project both vision backbones into hdim and sum -> add the 2D coordinate
    PE -> `pre_attn` (self-attn over all cameras' patches, cross-attn to language, with
    camera-pose PRoPE) -> QFormer compresses Ncam*Lv patches to 64 queries -> `post_attn`.

    The compression is not optional: with Ncam cameras at Lv patches each, feeding the
    raw tokens to every denoising step of the head would dominate inference cost.

    Three positional encodings do three different jobs here:
      * `proj_pe(norm_xy)` -- additive, absolute position on the image plane.
        zero-initialised so it does not perturb the frozen features early in training.
      * PRoPE(extrinsics)  -- multiplicative, applied inside attention; makes attention
        between two patches depend on the *relative* pose of their cameras.
      * `main_cam_embed`   -- marks camera 0, the frame actions are expressed in.
    """

    def __init__(self, hdim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.proj_v = nn.ModuleList([
            simple_mlp([768, hdim, hdim], ln=True),  # for dinov2 vision embeds
            simple_mlp([768, hdim, hdim], ln=True),  # for siglip vision embeds
        ])
        self.proj_l = nn.ModuleList([
            simple_mlp([768, hdim, hdim], ln=True)  # for siglip language embeds
        ])
        self.proj_pe = simple_mlp([2, hdim, hdim], ln=True)  # for normalized coordinates

        self.main_cam_embed = nn.Parameter(torch.zeros(hdim))
        self.pre_attn = DiT(hdim, num_heads, num_layers//2, use_adaln=False, 
                            pe_type="prope")  # actually, it is not a DiT but self-cross attention module
        self.qformer = QFormerITM(hdim, num_heads, num_layers=1, num_queries=64)
        self.post_attn = FFWSelfAttentionLayers(hdim, num_heads, num_layers//2, use_adaln=False,
                                                bias=True, qk_norm=True, ffn_expansion=2)
        self.reset_parameters()

    def reset_parameters(self):
        init_xncoder(self.post_attn.num_layers, self.post_attn)
        # zero both weight and bias, otherwise pe2d is a random constant offset at init
        nn.init.zeros_(self.proj_pe[-1].weight)
        nn.init.zeros_(self.proj_pe[-1].bias)

    def forward(
        self,
        vl_obs: Dict[str, Tensor],
        vl_feature: Dict[str, Tensor], 
        fp16: bool
    ):
        """
        Args:
            vl_obs (Dict[str, Tensor]):
                - rgb: (B, To, ncam, 3, H, W)
                - norm_xy: (B, To, ncam, 2, H, W), coordinates in normalized camera plane
                - text: List (length=B) of prompt
                - extrinsics: (B, To, ncam, 4, 4), ^{world}_{camera} T
            
            vl_feature (Dict[str, Tensor]):
                - norm_xy_ds: (B, Ncam, Lv, 2)
                - vision_embeds: List (length=num_layer) of (B, Ncam, Lv, C)
                - lang_embeds: List (length=num_layer) of (B, La, C)
                - lang_mask: (B, La)
                - extrinsics: (B, Ncam, 4, 4)

            fp16: if True, use bfloat16
        
        Returns
        -------
            context: (B, Ncam*Lt, hdim)
            context_mask: (B, Ncam*Lt)
        """
        batch_size, _, num_cam, _, _, _ = vl_obs["rgb"].shape
        obs_extrinsics = vl_obs["extrinsics"]  # (B, To, Ncam, 4, 4)

        # Rebase every camera pose onto camera 0 at the latest timestep. PRoPE only ever
        # uses relative poses, so the absolute world origin must not leak in -- otherwise
        # the model overfits to each dataset's arbitrary world frame.
        cam0_extr_ref = torch.inverse(obs_extrinsics[:, -1:, 0:1]) @ obs_extrinsics  # (B, To, Ncam, 4, 4)
        x_v: Tensor = self.proj_v[0](vl_feature["vision_embeds"][0]) + \
                      self.proj_v[1](vl_feature["vision_embeds"][1])  # (B, Ncam, Lv, C)
        x_l: Tensor = self.proj_l[0](vl_feature["lang_embeds"][0])    # (B, Ncam, La, C)
        norm_xy_ds = vl_feature["norm_xy_ds"]  # (B, Ncam, Lv, 2)
        mask_l = vl_feature["lang_mask"]  # (B, La)

        # camera pose as multiplicative positional encoding (PRoPE)
        num_patch = x_v.shape[-2]
        extrinsic_wcT = cam0_extr_ref[:, -1]  # (B, Ncam, 4, 4), select the latest frame
        # Invert once here, on (B, Ncam, 4, 4). PRoPE needs the inverse for every query
        # token in every layer; inverting after the expand would redo the same Ncam
        # matrices Lv times per layer. se3_inverse is the analytic rigid-transform
        # inverse -- valid because cam0_extr_ref is a product of rigid transforms.
        extrinsic_cwT = se3_inverse(extrinsic_wcT)  # (B, Ncam, 4, 4)

        def expand_to_tokens(extr: Tensor):
            """(B, Ncam, 4, 4) -> (B, Ncam*Lv, 4, 4): every patch inherits its camera's
            pose. `expand` keeps this a view, so no Lv-fold memory blowup."""
            extr = extr[:, :, None, :, :].expand(batch_size, num_cam, num_patch, 4, 4)
            return rearrange(extr, "b n l r c -> b (n l) r c")

        extrinsic_pe = expand_to_tokens(extrinsic_wcT)
        extrinsic_pe_inv = expand_to_tokens(extrinsic_cwT)

        # flatten cameras into the token axis: attention runs over all views jointly
        x_v = rearrange(x_v, "b n l c -> b (n l) c")
        pe2d = rearrange(self.proj_pe(norm_xy_ds), "b n l c -> b (n l) c")

        # SA before qformer
        with torch.autocast(
            x_v.device.type,
            torch.bfloat16 if fp16 else torch.float32
        ):
            x_v: Tensor = self.pre_attn(
                x=x_v + pe2d,  # additive 2D positional encoding
                x_pe=extrinsic_pe,
                x_mask=None,  # every vision patch is valid
                conds=[x_l],
                cond_masks=[mask_l],
                films=None,
                x_pe_inv=extrinsic_pe_inv
            )

        # tag camera 0's tokens: actions live in its frame, so the head must be able to
        # tell which view defines "forward". clone() because pre_attn's output may be a
        # view and the next line writes in place.
        x_v = x_v.clone()
        num_patch_flat = x_v.shape[1] // num_cam
        x_v[:, :num_patch_flat] = x_v[:, :num_patch_flat] + self.main_cam_embed

        with torch.autocast(
            x_v.device.type, 
            torch.bfloat16 if fp16 else torch.float32
        ):
            query, x_l, _ = self.qformer(
                x_vision=x_v,
                mask_vision=None,
                x_text=x_l,
                mask_text=mask_l,
            )

            query = self.post_attn(
                query=query,
            )[-1]
        
        cond = torch.cat([query, x_l], dim=1)
        cond_mask = concat_mask(mask0=None, mask1=mask_l,
                                L0=query.shape[1], L1=x_l.shape[1])
        return cond, cond_mask


class DiffusionHead(nn.Module):
    """DDIM epsilon predictor over a chunk of camera-frame end-effector actions.

    - Input: the encoded observation context and the noisy action chunk at step `t`
    - Output: the predicted noise, from which the scheduler derives step `t-1`

    Two different "times" run through this module; keep them apart:
      * `timestep`     -- the denoising step, a global condition driving adaLN/FiLM
      * chunk position -- where an action sits inside the horizon, an additive
                          sinusoidal embedding (`traj_time_embed`)

    NOTE on the history encoder: `hist_enc` is built for `action_dim - 1` inputs and is
    fed `history[..., :action_dim-1]`, i.e. the t3r6 pose delta only. Gripper openness
    (the last channel) is deliberately withheld from history so the model cannot simply
    copy the previous gripper command; it has to read openness off the wrist camera.

    NOTE on normalization: when `action_norm` is set, everything this module sees --
    `history`, `noisy_actions`, and the predicted noise -- lives in the *normalized*
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
        # predict exactly zero noise at init.
        if self.abs_pos_enc is not None:
            nn.init.zeros_(self.abs_pos_enc[-1].weight)
            nn.init.zeros_(self.abs_pos_enc[-1].bias)
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
        """One denoising step.

        History and the noisy action chunk are concatenated into a single sequence so
        self-attention can relate the two; only the action segment is read out at the
        end. History acts as clean, always-available conditioning.

        Args:
            timestep: (B,), denoising step (NOT the position within the chunk)
            noisy_actions: (B, Ta, action_dim), the noisy action chunk
            cur_wcT: (B, 4, 4), ^{world} T _{cam}
            cur_weT: (B, 4, 4), ^{world} T _{ee}
            history: (B, nhist, action_dim), past actions in the same encoding
            conds: [(B, Lc, hdim)], observation context from ContextEncoder
            cond_masks: [(B, Lc)] or [None]
            fp16 (bool): use bfloat16 autocast for the attention stack

        Returns:
            pred_noise: (B, Ta, action_dim)
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
        pred_noise = self.act_head(action_feats)
        return pred_noise


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
    ):
        super().__init__()
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

        self.context_encoder = ContextEncoder(hdim, num_heads, num_layers=num_context_layers)
        self.dp_head = DiffusionHead(hdim, num_heads, self.action_space,
                                     num_layers=num_diffusion_layers,
                                     action_norm=action_norm)

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=diffusion_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=False
        )

        self.diffusion_timesteps = diffusion_timesteps
        if inference_timesteps is None:
            inference_timesteps = max(diffusion_timesteps//5, 10)
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

    def iterative_denoise(
        self,
        actions_shape: Tuple[int, int, int],
        fixed_inputs: Dict[str, Tensor],
        initial_noise: Optional[Tensor] = None
    ):
        """Full DDIM reverse loop, from pure noise to an action chunk.

        `fixed_inputs` is everything that does not change across denoising steps (the
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
            vl_obs (Dict[str, Tensor]):
                - rgb: (B, To, ncam, 3, H, W)
                - norm_xy: (B, To, ncam, 2, H, W), coordinates in normalized camera plane
                - text: List (length=B) of prompt
                - extrinsics: (B, To, ncam, 4, 4), ^{world}_{camera} T
            
            vl_feature (Dict[str, Tensor]):
                - norm_xy_ds: (B, Ncam, Lv, 2)
                - vision_embeds: List (length=num_layer) of (B, Ncam, Lv, C)
                - lang_embeds: List (length=num_layer) of (B, La, C)
                - lang_mask: (B, La)
                - extrinsics: (B, Ncam, 4, 4)

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
        # camera 0 at the latest timestep is the reference frame for the whole action
        # representation -- ContextEncoder builds its PRoPE relative to the same camera
        latest_cam_poses = vl_obs["extrinsics"][:, -1]  # (B, Ncam, 4, 4)
        current_cam_pose = latest_cam_poses[:, 0]  # first camera, (B, 4, 4)

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
            pred_actions = self.iterative_denoise(
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

        pred_noise = self.dp_head(timesteps, noisy_actions, **fixed_inputs)
        target = get_target(future_action_cam, noise, timesteps, self.noise_scheduler)
        
        # # drop too aggresive actions
        # debug_gt_ee_pose = future_actions.transpose(1, 2)[valid_ee_mask][..., :16].reshape(flat_batch_size, -1, 4, 4)
        # debug_gt_ee_pos = debug_gt_ee_pose[..., :3, 3]  # (B', Ta, 3)
        # delta_norm = (debug_gt_ee_pos[:, 1:] - debug_gt_ee_pos[:, :-1]).norm(dim=-1)
        # debug_mask = delta_norm < 0.2  # (B', Ta-1)
        # debug_mask[..., 1:-1] = debug_mask[..., 1:-1] & debug_mask[..., :-2] & debug_mask[..., 2:]
        # debug_mask = torch.cat([debug_mask[:, 0:1], debug_mask], dim=-1)  # (B', Ta)
        # # filter
        # pred_noise = pred_noise[debug_mask]
        # target = target[debug_mask]

        # Loss is split per action channel only so the terms can be weighted and logged
        # separately -- with prediction_type="epsilon" every slice is part of the same
        # standard normal, so the weights are pure loss shaping. The split itself is a
        # property of the encoding, hence it lives on the action space.
        total_loss, metrics = self.action_space.loss(pred_noise, target)
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
    """Supervision target matching the scheduler's parameterisation.

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
    model = ActionExpert(
        hdim=256,
        num_heads=4,
        num_context_layers=8,
        num_diffusion_layers=4,
        diffusion_timesteps=100,
    )

    modules = [
        model
    ]

    num_param = 0
    for m in modules:
        for p in m.parameters():
            if not p.requires_grad:
                continue
            
            num_param += p.numel()

    print("[INFO] Total {:.3f}M trainable parameters"
          .format(num_param / 1e6))


if __name__ == "__main__":
    count_parameters()

