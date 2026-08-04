"""Compute q01/q99 action statistics for `TrainConfig.action_norm_stats`.

    python -m data_prepare.compute_action_stats --config finetune_libero_10 \
        -o ./action_stats/libero_10.json

WHAT IT MEASURES, and why it cannot be done by reading the HDF5 files directly: the
model does not act in the dataset's coordinates. It predicts a delta from the current
end-effector pose, expressed in the orientation of camera 0 at the latest observed
timestep, as 3 translation + a 6D rotation, plus a rescaled gripper openness. That
encoding depends on the DataConfig (which cameras, `sample_state_gaps`, the future
horizon, whether cameras are shuffled), so the statistics are a property of the
*config*, not of the dataset on disk. This script therefore builds the very same
datasets `train.py` would, draws samples through the very same `DataSampler`, and runs
the very same `states2action` -- with normalization off, since that is what it is
measuring.

The one deviation from the training path is `skip_rgb`: frames are not decoded, because
the action space does not depend on pixels and decoding is essentially all of the
runtime. Everything that touches poses, timing and alignment is untouched.

HISTORY IS INCLUDED by default. `history_actions` is fed through the same normalizer as
the predicted chunk (it is conditioning for the same head), so the quantiles must cover
both or clipping would silently truncate the history channel. Pass --future-only to
measure the predicted chunk alone.

Sampling is random-window (`latest=False`), matching training, so the number of samples
to draw is a real choice: --num-samples defaults to 20 windows per trajectory, which for
a few hundred demonstrations is tens of thousands of action vectors -- far more than the
percentile estimate needs.
"""

import os
import sys
import json
import argparse
from typing import List

import numpy as np
import torch

from configs import CONFIGS, TrainConfig
from models.action_norm import ActionNormalizer
from models.action_space import build_action_space
from data_utils.dataset_base import get_dataloader, generate_sample_weights


# names for the report only. The EE-pose space has 10 fixed channels; a joint space is
# nq joints + openness, so its names are generated.
EE_CHANNEL_NAMES = ["tx", "ty", "tz",
                    "r00", "r10", "r20", "r01", "r11", "r21",
                    "openness"]


def channel_names(action_space):
    if action_space.layout == "cam_rel_t3r6_openness":
        return EE_CHANNEL_NAMES
    return ["q{}".format(i) for i in range(action_space.action_dim - 1)] + ["openness"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, required=True,
                        help="name of a preset in configs.CONFIGS, e.g. finetune_libero_10")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="where to write the stats json")
    parser.add_argument("--num-samples", type=int, default=-1,
                        help="total windows to draw; <0 means 20 per trajectory")
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--future-only", action="store_true",
                        help="measure only the predicted chunk, excluding history")
    parser.add_argument("--no-clip", action="store_true",
                        help="write clip=false, so normalization never clamps to [-1,1]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu",
                        help="states2action is small; cpu is usually fastest end to end")
    return parser.parse_args(argv)


def build_datasets(cfg: TrainConfig):
    datasets = [D.inst() for D in cfg.dataset_classes]
    if not datasets:
        raise ValueError(
            "config has no dataset_classes -- nothing to compute statistics over. "
            "The 'debug' preset is empty by design; pick a real preset.")
    for d in datasets:
        # geometry only; see data_utils/h5io.py:slice_encoded_frames
        d.skip_rgb = True
    return datasets


@torch.no_grad()
def collect_actions(cfg: TrainConfig, args) -> np.ndarray:
    """Draw windows and map them into the model's action space.

    Returns:
        (N, action_dim) array of raw (unnormalized) actions in the configured space
    """
    datasets = build_datasets(cfg)
    num_traj = sum(len(d) for d in datasets)
    target = args.num_samples if args.num_samples > 0 else 20 * num_traj
    print("[INFO] {} trajectories across {} dataset(s); target {} windows"
          .format(num_traj, len(datasets), target))

    if cfg.dataset_weights is not None:
        sample_weights = generate_sample_weights(datasets, cfg.dataset_weights)
        # Draw with the training mixture weights, so a dataset that training sees 10x as
        # often also dominates the quantiles 10x as much. multiplex is chosen to reach
        # `target` draws; the loader is stopped early once we have enough.
        multiplex = max(1, int(np.ceil(target / max(1, len(sample_weights)))))
    else:
        sample_weights, multiplex = None, 1

    loader = get_dataloader(
        datasets=datasets,
        batch_size=args.bs,
        num_workers=args.workers,
        shuffle=True if sample_weights is None else None,
        persistent_workers=False,
        sample_weights=sample_weights,
        sample_multiplex=multiplex,
    )

    device = torch.device(args.device)
    # exactly the encoding the model will train in; measuring any other one is measuring
    # the wrong space
    action_space = build_action_space(cfg.action_space)
    chunks: List[np.ndarray] = []
    seen = 0

    for batch in loader:
        extrinsics = batch["obs_extrinsics"].to(device)     # (B, To, ncam, 4, 4)
        ee_poses = batch["ee_poses"].to(device)             # (B, Nee, 4, 4)
        valid_ee_mask = batch["valid_ee_mask"].to(device)   # (B, Nee)

        # exactly the reference frame ActionExpert.forward uses: camera 0, latest frame
        current_cam_pose = extrinsics[:, -1][:, 0]  # (B, 4, 4)

        valid_ee_per_batch = valid_ee_mask.sum(dim=-1)
        batch_index = torch.cat([torch.empty(n, dtype=torch.long).fill_(b)
                                 for b, n in enumerate(valid_ee_per_batch.tolist())]
                                ).to(device)

        keys = ["future_actions"] if args.future_only else \
               ["history_actions", "future_actions"]
        for key in keys:
            states = batch[key].to(device)  # (B, T, Nee, state_dim)
            action = action_space.states2action(
                current_cam_pose[batch_index],
                ee_poses[valid_ee_mask],
                states.transpose(1, 2)[valid_ee_mask],
                None,  # measuring the raw space is the whole point
            )  # (B', T, action_dim)
            chunks.append(action.reshape(-1, action.shape[-1]).float().cpu().numpy())

        seen += extrinsics.shape[0]
        if seen % (args.bs * 20) < args.bs:
            print("[INFO] {}/{} windows".format(seen, target), flush=True)
        if seen >= target:
            break

    if not chunks:
        raise RuntimeError("no samples were drawn -- is the dataset path list empty?")

    actions = np.concatenate(chunks, axis=0)
    print("[INFO] collected {} action vectors from {} windows"
          .format(len(actions), seen))
    return actions


def report(actions: np.ndarray, q01: np.ndarray, q99: np.ndarray, names: List[str]):
    print("\n{:>9} {:>12} {:>12} {:>12} {:>12} {:>12}"
          .format("channel", "min", "q01", "q99", "max", "span"))
    for i, name in enumerate(names[:actions.shape[1]]):
        span = q99[i] - q01[i]
        flag = "  <-- near-constant, will pass through unscaled" if span < 1e-6 else ""
        print("{:>9} {:>12.5f} {:>12.5f} {:>12.5f} {:>12.5f} {:>12.5f}{}"
              .format(name, actions[:, i].min(), q01[i], q99[i],
                      actions[:, i].max(), span, flag))

    outside = np.mean((actions < q01) | (actions > q99), axis=0)
    print("\nfraction outside [q01, q99] per channel (≈2% by construction; much more "
          "means a heavy tail that clipping will bite into):")
    print("  " + "  ".join("{}={:.3f}".format(n, f)
                           for n, f in zip(names, outside)))


def main(argv=None):
    args = parse_args(argv)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.config not in CONFIGS:
        raise SystemExit("unknown config '{}'. Available: {}"
                         .format(args.config, ", ".join(sorted(CONFIGS))))
    cfg = CONFIGS[args.config]

    action_space = build_action_space(cfg.action_space)
    print("[INFO] action space: {} (layout '{}', action_dim={})"
          .format(action_space.name, action_space.layout, action_space.action_dim))
    actions = collect_actions(cfg, args)
    q01 = np.percentile(actions, 1, axis=0)
    q99 = np.percentile(actions, 99, axis=0)
    report(actions, q01, q99, channel_names(action_space))

    normalizer = ActionNormalizer(
        q01=q01.tolist(),
        q99=q99.tolist(),
        clip=not args.no_clip,
        layout=action_space.layout,
        meta={
            "config": args.config,
            "layout": action_space.layout,
            "action_space": action_space.name,
            "datasets": [D.__name__ for D in cfg.dataset_classes],
            "dataset_weights": cfg.dataset_weights,
            "num_action_vectors": int(len(actions)),
            "includes_history": not args.future_only,
            "command": " ".join(sys.argv),
            # not used by the model; kept because "is this channel degenerate or did I
            # sample too few windows?" is the first question you ask of a bad stats file
            "q50": np.percentile(actions, 50, axis=0).tolist(),
            "min": actions.min(axis=0).tolist(),
            "max": actions.max(axis=0).tolist(),
        },
    )
    normalizer.to_json(args.output)
    print("\n[INFO] wrote {}".format(args.output))
    print("[INFO] use it with: --action_norm_stats {}".format(args.output))


if __name__ == "__main__":
    main()
