# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

E2VLA: a diffusion-transformer VLA. Frozen SigLIP + DINOv2 + SigLIP-text → `ContextEncoder`
(multi-view attention with camera-pose PE → QFormer bottleneck) → DDIM `DiffusionHead` that
denoises a chunk of end-effector actions. Pretrain on DROID/ManiSkill/MetaWorld, fine-tune +
evaluate on LIBERO, or fine-tune on your own robot.

`README.md` is the long-form manual (Chinese, ~600 lines) and is *current* — sections §3.5–§3.7
document LoRA, action normalization and the joint/EE action-space split in detail. Code comments
are a mix of English and Chinese; match the file you are editing.

---

## Commands

Everything runs from `e2vla/` — dataset classes glob relative paths (`./data_converted/...`),
so a different cwd yields "no episodes found". GPU selection is `CUDA_VISIBLE_DEVICES`; the code
then hardcodes `cuda:0`.

```bash
python train.py -h                    # prints BOTH argparse (--config/-s/-c) and tyro help
python datavis.py -l                  # list dataset classes
python datavis.py Libero10            # overlay ee axes + future traj — verify BEFORE training

# train: --config picks a preset from CONFIGS, tyro then overrides any TrainConfig field
CUDA_VISIBLE_DEVICES=0 python train.py --config pretrain -s EXP
CUDA_VISIBLE_DEVICES=0 python train.py --config finetune_libero_10 \
  --pretrained_ckpt ./checkpoints/E2VLA/PRETRAIN_EXP/ckpt_xxxxxxx.pt -s FT_EXP
CUDA_VISIBLE_DEVICES=0 python train.py --config finetune_libero_10 -c FT_EXP   # resume

# q01/q99 action stats (optional, off by default) — recompute per config, not per dataset
python -m data_prepare.compute_action_stats --config finetune_libero_10 -o ./action_stats/libero_10.json
```

Logs → `./logs/E2VLA/EXP` (TensorBoard), checkpoints → `./checkpoints/E2VLA/EXP/`
(`ckpt_latest.pt`, `ckpt_best.pt`, `ckpt_{iter:07d}.pt`, plus a `{launchtime}.json` config dump
that `infer_utils/planner.py` reads back at inference).

### Data conversion — separate conda envs

Each converter needs its own env because the source SDKs conflict; all write HDF5 into
`./data_converted/`, with `./data_raw/` symlinks to the originals. See README §1.

| dataset | env | script |
| --- | --- | --- |
| DROID (`cadene/droid_1.0.1`, has extrinsics) | `lerobot` (0.1.0) | `data_prepare/process_droid.py` |
| ManiSkill (RLDS/tfrecord) | `tensorflow` | `data_prepare/process_maniskill.py` |
| MetaWorld | `metaworld-v3` | `data_prepare/process_metaworld.py` |
| LIBERO | `libero` | `data_prepare/process_libero.py` |

`process_libero.py` is the reference implementation to copy when converting your own recordings.

### Evaluation — three processes, three terminals

Pyro4 RPC (`shm_transport/`) decouples the policy from the simulator, which needs a different env.

```bash
pyro4-ns                                                    # 1: nameserver on :9090
CUDA_VISIBLE_DEVICES=0 python -m infer_utils.remote_service \
  --ckpt ./checkpoints/E2VLA/FT_EXP/ckpt_xxxxxxx.pt --uri MY_URI [--ema] [--ensemble 3]
python -m examples.libero.eval --task_suite libero_10 --uri MY_URI --save --video
```

Pass `--ema` only if the run actually enabled EMA. LoRA needs no flag — ranks and targets are read
out of the checkpoint. Results → `./eval_results/`, videos → `./eval_videos/`.

### Self-tests (no test framework; each is a `__main__` block)

```bash
python -m models.vla                  # parameter counts for vla_base
python -m models.action_expert        # parameter count for the action expert alone
python -m models.action_norm          # normalize/unnormalize round-trip
python -m data_utils.dataset_real     # __getitem__ output-contract + unit checks (see below)
python -m models.encoders.siglip
```

`python -m data_utils.dataset_real` is the one worth running when wiring up new data: besides
shapes it checks that `rgbs` was divided by 255, that the gripper is in `[0,1]`, and — branching
on action space — that pose rows end in `[0,0,0,1]` or that joint angles are radians not degrees.
Degrees vs radians never errors during training, it just makes the normalization range 57× wide.

---

## Architecture

`models/vla.py:VLA = VLM + ActionExpert`, instantiated by `vla_tiny/small/base` (`base` = hdim 768,
102.3M trainable). Everything is threaded through two dicts, `vl_obs` and `vl_feature`.

**`models/vlm.py`** — SigLIP (`google/siglip-base-patch16-256`, vision + text) and DINOv2
(`dinov2_vitb14_reg` via `torch.hub`, so `HF_ENDPOINT` does *not* apply to it — see the class
docstring for the offline path). Both are frozen and `VLM.train()` pins them to eval. Only the
latest frame is used (`[:, -1]`) even though tensors carry a history axis.

**`models/action_expert.py`** — two modules and the action-encoding functions:
- `ContextEncoder` (~60% of params): projects both vision towers into `hdim` and sums → additive
  2D PE (`proj_pe`, zero-init) → `pre_attn` self-attn over *all cameras' patches jointly*,
  cross-attn to language, with **PRoPE** (multiplicative camera-pose PE) → QFormer compresses to
  64 queries → `post_attn`. Camera poses are rebased onto camera 0 at the latest timestep so the
  world origin never leaks in; `main_cam_embed` tags camera 0's tokens.
- `DiffusionHead`: DDIM epsilon prediction (`diffusion_timesteps=100`, 20 inference steps). Two
  distinct "times" — the denoising step (adaLN/FiLM condition) and the position inside the chunk
  (additive sinusoidal). History and the noisy chunk are concatenated into one sequence; only the
  chunk is read out. Gripper openness is deliberately withheld from history so the policy must
  read it off the wrist camera.

**Interface convention**: `(B, To, ncam, ...)` observations, `(B, Ta, Nee, state_dim)` actions.
Valid end-effectors are flattened `(B, Nee) → B'` and each becomes an independent sample sharing
its batch element's context.

**`models/layers/`** is the shared attention stack: `attn_dn.py` (`AdaLN`/`NormOrAdaLN`,
`CrossAttentionLayer`/`SelfAttentionLayer` and their `FFW*Layers` stacks, `init_xncoder` DeepNorm
init), `mha.py`, `pe.py` (`SinusoidalPosEmb` / `RoPE` / `PRoPE` / `se3_inverse`), `norms.py`,
`rot_transforms.py`, `utils.py`. `dit.py` composes them into `DiTBlock` (self-attn → cross-attn →
FFN, adaLN driven by FiLM). Positional encoding is selected by a `pe_type` string threaded down to
`mha.py`; `"prope"` needs camera extrinsics and `head_dim % 4 == 0`.

**`data_utils/dataset_base.py`** is the heavy part: `DataSampler` (temporal alignment,
history/future windows, camera padding) + `DataConfig` + `H5DatasetMapBase`. `data_utils/datasets.py`
declares one class per dataset (each with its own `DataConfig` and an `inst()` classmethod that
globs files) and exports `DATA_CONFIGS` keyed by **class name** — that name is what
`TrajPlanner.set_config` takes at inference time, and what `TrainConfig.dataset_classes` round-trips
through when a config is dumped to / loaded from JSON.

**`infer_utils/planner.py:TrajPlanner`** — `set_config` → `reset` → `set_prompt` →
`add_obs_frame` → `get_action` → `ensemble_traj`. `reset()` between episodes is mandatory: the
ensembler is indexed by timestamp and a new episode restarts at 0.

**`models/va_policy_example.py`** is a research bridge, not part of the training path: a
single-task VA policy (`StateEncoder`/`ActionDecoder`, plus `ActionDecoderMamba`) written in
e2vla's idiom for the sibling `pvrobo` project. It imports `.unet_mamba` and `.conditional_unet1d`,
which do not exist here — copy `pvrobo/src/agent/unet_mamba.py` in to run it.

---

## Things that fail silently

This codebase's central design concern is that a wrong checkpoint or a wrong frame **loads cleanly
and then produces garbage**. Several stamps exist purely to make those cases loud; don't remove or
work around them.

- **`envars.py` must be imported before torchvision.** It pokes `cv2.namedWindow` to dodge the
  av/opencv deadlock, and sets `HF_ENDPOINT` to the hf-mirror.
- **`camera_names[0]` defines the entire action frame.** `space_ee2cam` expresses every action in
  its coordinates and `main_cam_embed` tags its tokens. Third-person camera first, wrist second —
  matching pretraining (LIBERO `agentview`/`eye_in_hand`, DROID `exterior`/`wrist`).
- **Extrinsics are load-bearing under `ee_cam`** and their absence is not an error. Without
  calibration, fill `obs_extrinsics` with identity for the *whole* episode (degenerates to base
  frame, self-consistent) and set `shuffle_cameras=False`, never a mix of real and fake. `pe_type`
  is parameter-free, so switching it does not change the `state_dict` — a checkpoint still
  "loads", the PE just means something else and training stops converging.
- **Four stamps guard checkpoint compatibility** (`train_utils/ckpt.py`), all for the same reason:
  the offending checkpoint has identical tensor names and shapes.
  - `objective` (`"ddim"` / `"flow"`, set by `TrainConfig.objective`) — the same network under two
    generative objectives, so they share every tensor name and shape. Raises unconditionally; the
    deliberate cross-objective path is `pretrained_ignore_objective`, which transfers the trunk (96.5%
    of the expert) and re-zeroes `act_head`'s output Linear. Released pretrain checkpoints predate the
    stamp, so a *missing* key means DDIM.
  - `action_layout` (`cam_rel_t3r6_openness` / `abs_joint7_openness`) — two action spaces can have
    the same channel count (a 9-axis arm's joint space is also 10-dim).
  - `action_norm` — a checkpoint trained on normalized actions loads into an un-normalized model
    with zero missing keys and executes actions off by an affine.
  - `vlm_lora_rank` / `vlm_lora_targets` — the backbones are never serialized (rebuilt from
    HF/torch.hub each run), so these factors are the *only* record that the features the action
    expert saw are not the stock ones.
- **LoRA injection order is fixed and asymmetric**: fine-tune = load then inject (released
  checkpoints are clean); resume `-c` = inject then load (the checkpoint already has LoRA keys);
  deploy = inject → load → EMA `copy_to` → merge *last* (the EMA shadow *is* the LoRA factors).
  `lora_rank > 0` requires `--pretrained_ckpt`, else 60% of the model is frozen at random init.
  Action-expert LoRA is merged at deploy (zero overhead); VLM LoRA is deliberately **not** merged —
  bf16 rounding would swallow the adaptation.
- **Two `lora` knobs, different scopes**: `lora_rank` adapts `ContextEncoder` attention projections
  (+ LayerNorm affines, biases, `qformer.queries`, `main_cam_embed`); `vlm_lora_rank` adapts the
  frozen ViTs themselves and is the most expensive switch in the repo — configure it from a preset
  in `configs.py`, halve `bs`, and expect any frozen-feature cache to become invalid.
- **Action stats do not transfer.** They are computed over the *model's* action space, so they
  depend on the config (action space, cameras, sampling gaps, horizon), not on the data on disk.
  Recompute per fine-tuning set. Clipping deliberately skips the 6D rotation channels
  (`DEFAULT_CLIP_DIMS = (0, 1, 2, 9)`): clipping them can collapse the two 3-vectors to collinear,
  and Gram-Schmidt then returns a matrix with a zero row.
- **`joint*` action space is training-only.** `TrajPlanner` decodes 17-dim SE(3) exclusively
  (`_make_empty_action`, `get_action`'s `reshape(..., 4, 4)`, `ensemble_traj`), so a joint
  checkpoint builds fine in `remote_service` and then crashes at the reshape. Crossing spaces from
  an `ee_cam` pretrain checkpoint needs *both* `--pretrained_strict False` and
  `--pretrained_ignore_action_layout True` (99.98% of weights transfer; only `dp_head.hist_enc.0`,
  `dp_head.traj_enc.0`, `dp_head.act_head.3` are re-initialised, and `abs_pos_enc` is not built at
  all because recovering ee position from joints needs forward kinematics).

---

## Repo state

This is its own git clone (`git -C e2vla ...`); `VA/` above it is a workspace, not a repo. HEAD is
usually not what's running — check `git status` first, uncommitted work tends to sit in
`models/layers/`, `train_utils/lora.py` and `models/vlm.py`.
