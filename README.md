# e2vla

Project under refactoring

Checkpoints, evaluation results and videos have been uploaded to google drive:
* Checkpoints: [link](https://drive.google.com/drive/folders/1rMpj7ry4YObLciNdPjY9DiZ-JpT7uG_q?usp=sharing)
* LIBERO scores: [link](https://drive.google.com/drive/folders/1RxZPSohDQVi2vH6zYSujVJxuPdOwt29y?usp=sharing)
* LIBERO videos: [link](https://drive.google.com/drive/folders/1bcp7wB13i3HoRLd2Xe1sHtqVVuSzkGRj?usp=sharing)


# Dependency


# Pretrain
## 1. Dataset Preparation
* Droid
  
  We use the processed data from [cadence/droid_1.0.1](https://huggingface.co/datasets/cadene/droid_1.0.1) as it has camera extrinsic attached. Download it to anywhere you like, and make a symbolic link to it as `./data_raw/droid_1.0.1`. Then run:
  ```bash
  conda activate lerobot
  python data_prepare/process_droid.py \
    --input_root ./data_raw/droid_1.0.1 \
    --alter_vid_root VIDEO_DOWNLOAD_PATH \
    --output_root ./data_converted/droid \
    --skip_saved
  ```
  **Note:**
  * This requires [lerobot](https://github.com/huggingface/lerobot) installed. We use version 0.1.0. You may need to create a new conda environment (e.g. named `lerobot`) and install the package via:
    ```bash
    pip install "lerobot==0.1.0"
    ```
  * The initial downloads of video files may be incomplete (test at 2025/04). We need to download the full video files and place them at `VIDEO_DOWNLOAD_PATH`. **TODO:** upload scripts to fix this.
  
* Maniskill
  
  First download the [data](https://www.tensorflow.org/datasets/catalog/maniskill_dataset_converted_externally_to_rlds) to anywhere you like, e.g.:
  ```bash
  mkdir -p ANYWHERE/maniskill
  gsutil -m cp -r gs://gresearch/robotics/maniskill_dataset_converted_externally_to_rlds/0.1.0 ANYWHERE/maniskill
  ln -s ANYWHERE/maniskill ./data_raw/maniskill
  ```
  Then run:
  ```bash
  conda activate tensorflow
  python data_prepare/process_maniskill.py \
    --input_root ./data_raw/maniskill/0.1.0 \
    --output_root ./data_converted/maniskill/0.1.0 \
    --visualize
  ```
  Note:
  * This requires [tensorflow](https://www.tensorflow.org/install) installed. You may need to create a new conda environment (e.g. named `tensorflow`) to install it and run the above command to generate data.
  
* Metaworld
  
  This doesn't require downloading extra data. However, you may still need to create a new conda environment (e.g. named `metaworld-v3`) and then install the [metaworld](https://github.com/Farama-Foundation/Metaworld) package via:
  ```bash
  pip install "metaworld==2.0.0"
  ```
  Then run:
  ```bash
  conda activate metaworld-v3
  python data_prepare/process_metaworld.py \
    --output_root ./data_converted/metaworld \
    --visualize \
    --skip_saved
  ```
  Note:
  * Although we install "metaworld==2.0.0", it is actually version 3.

If you have downloaded and processed all the data, the file structure would be like this: 
```
data_raw
├── droid_1.0.1
│   ├── README.md
│   ├── data
│   │   ├── chunk-000
│   │   ├── chunk-001
│   │   └── ...
│   ├── meta
│   │   ├── episodes.jsonl
│   │   ├── episodes_stats.jsonl
│   │   ├── info.json
│   │   └── tasks.jsonl
│   └── videos
│       ├── chunk-000
│       ├── chunk-001
│       └── ...
├── libero
│   ├── datasets
│   ├── libero_10
│   │   ├── KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5
│   │   ├── KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5
│   │   └── ...
│   ├── libero_90
│   │   ├── KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_demo.hdf5
│   │   ├── KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_demo.hdf5
│   │   └── ...
│   ├── libero_goal
│   │   ├── open_the_middle_drawer_of_the_cabinet_demo.hdf5
│   │   ├── open_the_top_drawer_and_put_the_bowl_inside_demo.hdf5
│   │   └── ...
│   ├── libero_object
│   │   ├── pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
│   │   ├── pick_up_the_bbq_sauce_and_place_it_in_the_basket_demo.hdf5
│   │   └── ...
│   └── ...
└── maniskill
    └── 0.1.0
        ├── dataset_info.json
        ├── features.json
        ├── maniskill_dataset_converted_externally_to_rlds-train.tfrecord-00000-of-01024
        ├── maniskill_dataset_converted_externally_to_rlds-train.tfrecord-00001-of-01024
        └── ...

data_converted
├── drawer
│   ├── 0000.h5
│   ├── 0001.h5
│   └── ...
├── droid
│   ├── data
│   │   ├── chunk-000
│   │   ├── chunk-001
│   │   └── ...
│   └── videos
│       ├── chunk-000
│       ├── chunk-001
│       └── ...
├── libero
│   ├── libero_10_no_noops
│   │   ├── KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
│   │   ├── KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it
│   │   └── ...
│   ├── libero_90_no_noops (not used in fine-tuning)
│   │   ├── KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet
│   │   ├── KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it
│   │   └── ...
│   ├── libero_goal_no_noops
│   │   ├── open_the_middle_drawer_of_the_cabinet
│   │   ├── open_the_top_drawer_and_put_the_bowl_inside
│   │   └── ...
│   ├── libero_object_no_noops
│   │   ├── pick_up_the_alphabet_soup_and_place_it_in_the_basket
│   │   ├── pick_up_the_bbq_sauce_and_place_it_in_the_basket
│   │   └── ...
│   └── libero_spatial_no_noops
│       ├── pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
│       ├── pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate
│       └── ...
├── maniskill
│   └── 0.1.0
│       ├── 00000.h5
│       ├── 00001.h5
│       └── ...
├── metaworld
│   ├── assembly-v3
│   │   ├── 0000.h5
│   │   ├── 0001.h5
│   │   └── ...
│   ├── basketball-v3
│   │   ├── 0000.h5
│   │   ├── 0001.h5
│   │   └── ...
│   └── ...
├── oven
│   ├── 0000.h5
│   ├── 0001.h5
│   └── ...
└── pick-place-can
    ├── 0000.h5
    ├── 0001.h5
    └── ...
```

## (1.5) Data Visualization
Visualize the processed data is recommended before training. Run:
```bash
python datavis.py {DATASET_NAME}
```
to visualize the specified dataset. Run `python datavis.py -l` to list all the available datasets.

## 2. Start Pre-training
You can use `python train.py -h` to see the help message. To pretrain on the above three datasets, run:
```bash
CUDA_VISIBLE_DEVICES=x python train.py --config pretrain -s EXPERIMENT_NAME
```
To pretrain on all the datasets mentioned in paper, run:
```bash
CUDA_VISIBLE_DEVICES=x python train.py --config pretrain_extra -s EXPERIMENT_NAME
```
This will save the log to `./logs/E2VLA/EXPERIMENT_NAME` and save the checkpoints to `./checkpoints/E2VLA/EXPERIMENT_NAME`.

We have uploaded two checkpoints [here](https://drive.google.com/drive/folders/1rMpj7ry4YObLciNdPjY9DiZ-JpT7uG_q?usp=sharing).

# Fine-tune and Evaluation on LIBERO
## 1. Dataset Preparation
First download the [LIBERO dataset](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets) to anywhere and then make a symbolink to `./data_raw/libero`. Then run 
```bash
conda activate libero
python data_prepare/process_libero.py \
  --libero_task_suite libero_spatial \
  --libero_raw_data_dir ./data_raw/libero \
  --libero_target_dir ./data_converted/libero \
  --skip_saved \
  --visualize
```
Change the libero_spatial to [libero_object, libero_goal, libero_10] for finetuning and evaluation on other task-suites.

## 2. Fine-tuning
For example, if we wnat to fine-tune on libero-10 from pretrained models:
```bash
CUDA_VISIBLE_DEVICES=x python train.py \
  --config finetune_libero_10 \
  --pretrained_ckpt ./checkpoints/E2VLA/PRETRAIN_EXP_NAME/ckpt_xxxxxxx.pt \
  -s FINETUNE_EXPERIMENT_NAME
```
This will load the config and the pre-trained weights. The fine-tuned weights are saved to `./checkpoints/E2VLA/FINETUNE_EXPERIMENT_NAME/`. We save the weights every 10k iterations by default.

Finetuned checkpoints can be found [here](https://drive.google.com/drive/folders/1rMpj7ry4YObLciNdPjY9DiZ-JpT7uG_q?usp=sharing).

## 3. Evaluation
* First we need to launch the pyro4 naming server (something like roscore). Open a separate terminal and run:
  ```bash
  pyro4-ns
  ```
  By default the naming server runs on `localhost:9090`.

* Launch planning service of your fine-tuned model:
  ```bash
  CUDA_VISIBLE_DEVICES=x python -m infer_utils.remote_service \
    --ckpt ./checkpoints/E2VLA/FINETUNE_EXPERIMENT_NAME/ckpt_xxxxxxx.pt \
    --uri CUSTOM_URI_NAME
  ```

* Start evaluation in simulation:
  ```bash
  python -m examples.libero.eval \
    --task_suite libero_10 \
    --uri CUSTOM_URI_NAME \
    --save --video
  ```
  The results are saved to `./eval_results/TASK_SUITE/URI/`, and the videos are saved to `./eval_videos/TASK_SUITE/URI/`

Evaluation results and videos using our fine-tuned checkpoints can be found [here](https://drive.google.com/drive/folders/1RxZPSohDQVi2vH6zYSujVJxuPdOwt29y?usp=sharing) and [here](https://drive.google.com/drive/folders/1bcp7wB13i3HoRLd2Xe1sHtqVVuSzkGRj?usp=sharing).

# Fine-tune on Own Data

This section covers fine-tuning on a real robot from a small number of demonstrations
(order 50-100 episodes, single robot, single task).

**Starting from a pretrained checkpoint is recommended, but it is not a hard
requirement** — single-task behaviour cloning from scratch on 50-100 demonstrations is
standard practice (Diffusion Policy, ACT), at comparable model sizes. Treat it as an
empirical question and, if you can afford two 20k-step runs, measure it.

What the pretrained checkpoint buys you here, concretely: 60% of the trainable
parameters live in the `ContextEncoder`, whose job is to fuse multiple camera views via
PRoPE and compress them through the QFormer bottleneck. Learning to exploit relative
camera geometry is the most data-hungry part of this architecture and the part that
transfers best — more so than the diffusion head itself. The pretraining mixture is
DROID-weighted (real Franka data with calibrated extrinsics), so a similar 7-DoF arm
with a third-person + wrist camera is close-domain.

Where pretrained init helps least: if your control frequency, gripper convention, or
camera arrangement differ substantially from the pretraining data, much of the prior
does not apply, and from-scratch with more iterations may be competitive.

Note that the vision backbones are frozen pretrained weights in either case, so
"from scratch" never means "from random visual features".

## 0. Prerequisites

**Calibrated camera extrinsics are mandatory, and their absence fails silently.**
The context encoder uses PRoPE, which consumes `^{world}_{cam}T`, and the action space
itself is defined in the camera frame (`space_ee2cam`). Note that PRoPE and RoPE are
both parameter-free, so switching `pe_type` does *not* change the `state_dict` — a
checkpoint would still "load successfully" while the positional encoding means
something entirely different. Symptom: training simply does not converge. Calibrate
your cameras before recording.

You need, for every frame:

| Quantity | Shape | Notes |
| --- | --- | --- |
| End-effector pose | `(4, 4)` | `^{world}_{ee}T`, homogeneous |
| Gripper openness | scalar | normalised to `[0, 1]`, 0 = closed, 1 = open |
| RGB per camera | `(H, W, 3)` | uint8 |
| Camera extrinsic | `(4, 4)` | `^{world}_{cam}T`, per frame (wrist cameras move) |
| Camera intrinsic | `(3, 3)` | pinhole `K`, constant per episode |
| Timestamp | scalar | seconds, monotonically increasing |

The world frame is arbitrary but must be *consistent within an episode* — the model
only ever uses poses relative to camera 0, so the origin never leaks into the network.

## 1. Convert your recordings to HDF5

One `.h5` file per episode, under `./data_converted/real_robot/`. Use
`data_prepare/process_libero.py` as the reference implementation. The layout, as
consumed by `DataSampler.sample_hdf5`:

```
episode_0001.h5
├── ee_pose            (T, 4, 4)  float32   ^{world}_{ee} T
├── gripper            (T,)       float32   [0 (closed), 1 (open)]
├── timestamp          (T,)       float32   seconds
├── ee_pose_desired    (T, 4, 4)  float32   optional: commanded pose (see note)
├── gripper_desired    (T,)       float32   optional: commanded gripper
├── exterior/                               <- group name = camera_names[0]
│   ├── rgb            (T, 3, H, W) uint8, or a vlen list of JPEG bytes
│   ├── pose           (T, 4, 4)  float32   ^{world}_{cam} T
│   └── K              (3, 3)     float32
└── wrist/                                  <- group name = camera_names[1]
    ├── rgb, pose, K   (same as above)
└── .attrs["prompt_text"] = "pick up the red block"
```

Notes:
- `ee_pose_desired` / `gripper_desired` are optional. If present they are used as the
  supervision target instead of the *achieved* pose, which is usually what you want on
  a real robot: the achieved pose lags the command, so training on it teaches the
  policy to under-shoot. Record your controller setpoints if you can.
- `prompt_text` is an HDF5 attribute, not a dataset. Multiple attributes whose name
  contains `prompt_text` are sampled from at random, which is a cheap way to add
  language augmentation. A single task still needs one, since the frozen SigLIP text
  encoder always runs.
- Images may be stored raw or JPEG-encoded; `transform_image` in
  `data_prepare/process_libero.py` shows both paths.

Verify the conversion *before* training — this catches extrinsic and gripper-convention
mistakes that are otherwise invisible in the loss:

```bash
python datavis.py -l                # list converted datasets
python datavis.py RealRobot         # overlays the ee axes and future trajectory
```

If the projected end-effector axes do not land on the real gripper in the overlay, your
extrinsics or intrinsics are wrong. Fix that first; nothing downstream will work.

## 2. Declare the dataset

Edit the four `TODO` fields in `RealRobot` in `data_utils/datasets.py`:

```python
class RealRobot(H5DatasetMapBase):
    config = DataConfig(
        sample_dt=1.0 / 15,                      # 1 / control_hz, wall-clock seconds
        record_dt=None,                          # None -> infer from `timestamp`
        camera_names=("exterior", "wrist"),      # third-person FIRST
        ee_indices=(0,),                         # single arm
        output_image_hw=(256, 256),
    )
```

`camera_names[0]` is not cosmetic: it defines the reference frame for the entire action
representation (`main_cam_embed` tags its tokens, `space_ee2cam` expresses every action
in its coordinates). Put the third-person camera first, matching the pretraining
datasets (LIBERO `agentview` then `eye_in_hand`; DROID `exterior` then `wrist`).

`sample_dt` must be the real wall-clock period. It sets how far ahead the predicted
chunk reaches: `sample_dt * sample_state_gaps * num_future_states` seconds — 2.13 s at
the defaults.

## 3. Fine-tune

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config finetune_real \
  --pretrained_ckpt ./checkpoints/E2VLA/PRETRAIN_EXP_NAME/ckpt_xxxxxxx.pt \
  -s MY_ROBOT_EXP
```

The `finetune_real` preset differs from the LIBERO ones in ways that all follow from the
smaller dataset: 20k iterations instead of 70k, `max_lr=5e-5` instead of `1e-4`,
`grad_clip=1.0`, `bs=16`, and EMA enabled from iteration 1k. `sample_multiplex=1000`
inflates the episode list so an "epoch" is a usable length.

Checkpoint loading behaviour:
- `--pretrained_ckpt` loads **weights only** by default (`pretrained_weights_only=True`).
  Fine-tuning is a new optimisation problem: `current_iters` resets to 0 and the LR
  re-warms from zero, so inheriting the pretrain run's Adam moments would make the first
  updates larger than intended. Pass `--pretrained_weights_only False` for the old
  behaviour.
- Loading is **strict** by default and reports exactly which tensors mismatch. If you
  see a shape mismatch under `context_encoder.proj_*` or `context_encoder.post_attn`,
  the checkpoint predates the `compact model` commit (122.4M vs the current 102.3M
  params). Either use a current checkpoint or pass `--pretrained_strict False` to load
  the compatible subset — the rest is then trained from scratch, which is stated loudly
  in the log rather than happening silently.
- `-c EXP_NAME` (resume) always restores the full optimiser state, unaffected by these
  flags.

## 4. Deploy

Same three-process setup as LIBERO, but your client feeds real observations instead of a
simulator. Start the nameserver and the policy server:

```bash
pyro4-ns
CUDA_VISIBLE_DEVICES=0 python -m infer_utils.remote_service \
  --ckpt ./checkpoints/E2VLA/MY_ROBOT_EXP/ckpt_xxxxxxx.pt \
  --uri MY_ROBOT --ensemble 3
```

Add `--ema` only if the run actually had EMA enabled (`finetune_real` does).

Your control loop then mirrors `examples/libero/eval.py`:

```python
controller = get_shm_proxy("MY_ROBOT", ns_host="localhost", ns_port=9090)
controller.set_config("RealRobot")     # keyed by the dataset class name
controller.reset()                     # REQUIRED between episodes
controller.set_prompt("pick up the red block")

while not done:
    controller.add_obs_frame(obs_frame)          # see TrajPlanner.add_obs_frame
    future_ee_poses, future_grippers, future_time, _ = controller.get_action()
    future_ee_poses  = future_ee_poses[:, 0]     # ee_indices == (0,)
    future_grippers  = future_grippers[:, 0]
    future_ee_poses, future_grippers = controller.ensemble_traj(
        future_ee_poses, future_grippers, future_time)
    # execute the first few waypoints, then re-plan
```

`controller.reset()` between episodes is not optional: the ensembler keys on timestamps,
which restart for a new episode, so stale chunks would otherwise be blended into the
first actions of the next one.

`obs_frame` uses the same schema as the HDF5 conversion — see the docstring of
`TrajPlanner.add_obs_frame` for the exact dict layout, and `obs_libero2ours` in
`examples/libero/eval.py` for a worked example.

