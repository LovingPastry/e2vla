# Repository Guidelines

## Project Structure & Module Organization

E2VLA is a Python vision-language-action research project. Commands must run from the repository root because datasets use relative paths.

- `models/`: VLA components, encoders, action spaces, and layers.
- `data_utils/`: dataset contracts, HDF5 access, alignment, and decoding.
- `data_prepare/`: converters for DROID, LIBERO, ManiSkill, and MetaWorld.
- `train_utils/` and `infer_utils/`: checkpointing, logging, planning, RPC, and visualization.
- `examples/libero/`: LIBERO rollout evaluation.
- `configs.py`, `train.py`, `test.py`: experiment presets, training, and open-loop evaluation.

Do not commit datasets, checkpoints, logs, or generated videos.

## Execution Environment

This checkout is for code editing; datasets and compute exist only on the cloud server. Do not treat missing local data, checkpoints, CUDA devices, or model caches as defects. Perform local static checks when possible, but run data conversion, visualization, training, and checkpoint evaluation on the server. If server execution is unavailable, state which runtime checks remain unverified.

## Build, Test, and Development Commands

There is no build step. Use the dataset-specific environments described in `README.md`.

```bash
python train.py -h
python datavis.py -l
CUDA_VISIBLE_DEVICES=0 python train.py --config pretrain -s EXP_NAME
CUDA_VISIBLE_DEVICES=0 python test.py --ckpt PATH --dataset Libero10
python -m data_utils.dataset_real
python -m models.action_norm
```

The first two inspect options and datasets. Training writes to `logs/E2VLA/` and `checkpoints/E2VLA/`; the final two run module self-tests.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and uppercase constants. Keep type hints on public interfaces. Comments may be English or Chinese; match the surrounding file. Derive related `configs.py` presets rather than duplicating them. No formatter or linter is configured; keep imports grouped and changes narrow.

## Testing Guidelines

The repository uses executable `__main__` checks rather than pytest and sets no coverage threshold. Run the self-test nearest your change; for data changes, prioritize `python -m data_utils.dataset_real`. Validate converted samples with `python datavis.py DATASET_NAME` before training. For model changes, run the relevant module (for example, `python -m models.vla vl`) and a small open-loop evaluation when possible.

## Commit & Pull Request Guidelines

Recent commits use short, topic-focused subjects such as `fix gripper range` and `ResNet Encoder`. Use an imperative summary, keep each commit single-purpose, and avoid committing generated artifacts. Pull requests should explain the behavioral change, list configurations and commands tested, link related issues or experiments, and include plots, metrics, or screenshots when outputs or visualizations change. Call out checkpoint or dataset compatibility changes explicitly.
