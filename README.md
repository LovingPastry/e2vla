# e2vla

项目重构中。

Checkpoint、评测结果和视频已上传至 Google Drive:
* Checkpoints: [link](https://drive.google.com/drive/folders/1rMpj7ry4YObLciNdPjY9DiZ-JpT7uG_q?usp=sharing)
* LIBERO scores: [link](https://drive.google.com/drive/folders/1RxZPSohDQVi2vH6zYSujVJxuPdOwt29y?usp=sharing)
* LIBERO videos: [link](https://drive.google.com/drive/folders/1bcp7wB13i3HoRLd2Xe1sHtqVVuSzkGRj?usp=sharing)


# 依赖


# 预训练
## 1. 数据准备
* Droid

  我们使用 [cadence/droid_1.0.1](https://huggingface.co/datasets/cadene/droid_1.0.1) 处理好的数据,因为它带有相机外参。下载到任意位置,然后软链接到 `./data_raw/droid_1.0.1`,再运行:
  ```bash
  conda activate lerobot
  python data_prepare/process_droid.py \
    --input_root ./data_raw/droid_1.0.1 \
    --alter_vid_root VIDEO_DOWNLOAD_PATH \
    --output_root ./data_converted/droid \
    --skip_saved
  ```
  **注意:**
  * 需要安装 [lerobot](https://github.com/huggingface/lerobot),我们使用 0.1.0 版本。可能需要新建一个 conda 环境(例如叫 `lerobot`)来安装:
    ```bash
    pip install "lerobot==0.1.0"
    ```
  * 初次下载的视频文件可能不完整(2025/04 测试)。需要下载完整视频文件并放到 `VIDEO_DOWNLOAD_PATH`。**TODO:** 上传修复脚本。

* Maniskill

  先把 [数据](https://www.tensorflow.org/datasets/catalog/maniskill_dataset_converted_externally_to_rlds) 下载到任意位置,例如:
  ```bash
  mkdir -p ANYWHERE/maniskill
  gsutil -m cp -r gs://gresearch/robotics/maniskill_dataset_converted_externally_to_rlds/0.1.0 ANYWHERE/maniskill
  ln -s ANYWHERE/maniskill ./data_raw/maniskill
  ```
  然后运行:
  ```bash
  conda activate tensorflow
  python data_prepare/process_maniskill.py \
    --input_root ./data_raw/maniskill/0.1.0 \
    --output_root ./data_converted/maniskill/0.1.0 \
    --visualize
  ```
  注意:
  * 需要安装 [tensorflow](https://www.tensorflow.org/install)。可能需要新建一个 conda 环境(例如叫 `tensorflow`)来安装并运行上述命令生成数据。

* Metaworld

  不需要额外下载数据。但可能仍需新建一个 conda 环境(例如叫 `metaworld-v3`),然后安装 [metaworld](https://github.com/Farama-Foundation/Metaworld):
  ```bash
  pip install "metaworld==2.0.0"
  ```
  然后运行:
  ```bash
  conda activate metaworld-v3
  python data_prepare/process_metaworld.py \
    --output_root ./data_converted/metaworld \
    --visualize \
    --skip_saved
  ```
  注意:
  * 虽然装的是 "metaworld==2.0.0",实际版本是 3。

如果所有数据都已下载并处理完毕,文件结构应该是这样:
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

## (1.5) 数据可视化
建议在训练前先可视化处理好的数据。运行:
```bash
python datavis.py {DATASET_NAME}
```
来可视化指定数据集。运行 `python datavis.py -l` 列出所有可用数据集。

## 2. 开始预训练
可以用 `python train.py -h` 查看帮助信息。在上述三个数据集上预训练:
```bash
CUDA_VISIBLE_DEVICES=x python train.py --config pretrain -s EXPERIMENT_NAME
```
在论文提到的所有数据集上预训练:
```bash
CUDA_VISIBLE_DEVICES=x python train.py --config pretrain_extra -s EXPERIMENT_NAME
```
日志保存到 `./logs/E2VLA/EXPERIMENT_NAME`(TensorBoard,写了什么见 §3.9),checkpoint 保存到 `./checkpoints/E2VLA/EXPERIMENT_NAME`。

我们上传了两个 checkpoint,见 [这里](https://drive.google.com/drive/folders/1rMpj7ry4YObLciNdPjY9DiZ-JpT7uG_q?usp=sharing)。

# 在 LIBERO 上微调与评测
## 1. 数据准备
先把 [LIBERO 数据集](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets) 下载到任意位置,软链接到 `./data_raw/libero`,然后运行:
```bash
conda activate libero
python data_prepare/process_libero.py \
  --libero_task_suite libero_spatial \
  --libero_raw_data_dir ./data_raw/libero \
  --libero_target_dir ./data_converted/libero \
  --skip_saved \
  --visualize
```
把 libero_spatial 换成 [libero_object, libero_goal, libero_10] 可在其他 task-suite 上微调和评测。

## 2. 微调
例如,要从预训练模型出发在 libero-10 上微调:
```bash
CUDA_VISIBLE_DEVICES=x python train.py \
  --config finetune_libero_10 \
  --pretrained_ckpt ./checkpoints/E2VLA/PRETRAIN_EXP_NAME/ckpt_xxxxxxx.pt \
  -s FINETUNE_EXPERIMENT_NAME
```
这会加载配置和预训练权重。微调后的权重保存到 `./checkpoints/E2VLA/FINETUNE_EXPERIMENT_NAME/`,默认每 10k 步保存一次。

微调好的 checkpoint 见 [这里](https://drive.google.com/drive/folders/1rMpj7ry4YObLciNdPjY9DiZ-JpT7uG_q?usp=sharing)。

## 3. 评测
* 首先启动 pyro4 命名服务(类似 roscore)。新开一个终端运行:
  ```bash
  pyro4-ns
  ```
  默认命名服务跑在 `localhost:9090`。

* 启动微调模型的规划服务:
  ```bash
  CUDA_VISIBLE_DEVICES=x python -m infer_utils.remote_service \
    --ckpt ./checkpoints/E2VLA/FINETUNE_EXPERIMENT_NAME/ckpt_xxxxxxx.pt \
    --uri CUSTOM_URI_NAME
  ```

* 在仿真中开始评测:
  ```bash
  python -m examples.libero.eval \
    --task_suite libero_10 \
    --uri CUSTOM_URI_NAME \
    --save --video
  ```
  结果保存到 `./eval_results/TASK_SUITE/URI/`,视频保存到 `./eval_videos/TASK_SUITE/URI/`。

用我们微调 checkpoint 得到的评测结果和视频见 [这里](https://drive.google.com/drive/folders/1RxZPSohDQVi2vH6zYSujVJxuPdOwt29y?usp=sharing) 和 [这里](https://drive.google.com/drive/folders/1bcp7wB13i3HoRLd2Xe1sHtqVVuSzkGRj?usp=sharing)。

# 用自己的数据微调

本节介绍如何用少量真机演示数据微调(约 50-100 条 episode,单机器人、单任务)。

## 先选一条路:末端位姿 还是 关节角

这是第一个决定,而且**决定之后不能中途改** —— 它改变 `action_dim`,`hist_enc` / `traj_enc` / `act_head` 全部换形状,没有 checkpoint 能跨过这条边界。

| | `action_space="ee_cam"`(默认) | `action_space="joint7"` |
| --- | --- | --- |
| 模型输出 | 相机系下的 SE(3) 增量 + 夹爪,10 维 | 绝对关节角 + 夹爪,nq+1 维 |
| 预训练权重 | **可用**,精确加载 | **可迁移 99.98%**,3 个边界层重新初始化 |
| 相机标定 | 必需(动作定义在相机系) | 不进入动作链路,只影响 PRoPE |
| 逆运动学 | 执行时需要 | 不需要,直接下发关节控制器 |
| 头部绝对位置编码 | 有 | 无(需要正运动学才能复原) |
| 数据格式 | HDF5(§1) | HDF5 或 memmap/bin(§1b) |
| 部署 | 已支持(§4) | **尚未支持**,见 §4 的说明 |
| LoRA | 可用 | 需先做跨空间迁移加载(§3.7) |

单任务真机上关节角常常更好用:动作直接是控制器的输入,标定误差不进入动作表示。代价不是"放弃预训练"——只有 3 个边界层(约 2.2 万参数)与动作编码绑定,99.98% 的权重照样迁移,详见 §3.7。真正放弃的是头部的绝对位置编码。走这条路请通读 §3.7。

**走 `ee_cam` 的话,推荐从预训练 checkpoint 出发,但这不是硬性要求** —— 在 50-100 条演示上做单任务行为克隆是标准做法(Diffusion Policy、ACT 都是这个量级的模型和数据)。把它当成一个实验问题:如果能跑两次 20k 步,直接测一下。

预训练 checkpoint 具体带来什么:可训练参数的 60% 在 `ContextEncoder` 里,它的职责是通过 PRoPE 融合多路相机视角,再经 QFormer 瓶颈压缩。学会利用相对相机几何是这个架构中最吃数据的部分,也是迁移性最好的部分 —— 比扩散头本身更值得迁移。预训练数据配比以 DROID 为主(带标定外参的真实 Franka 数据),所以一台类似的 7-DoF 机械臂 + 第三人称相机 + 腕部相机属于近域。

预训练收益最小的情况:如果你的控制频率、夹爪约定或相机布置与预训练数据差别很大,大部分先验就用不上了,从头训 + 增加迭代步数可能同样有竞争力。

注意视觉 backbone 在两种情况下都是冻结的预训练权重,所以"从头训"从来不意味着"从随机视觉特征开始"。

## 0. 前置条件

**`ee_cam` 下,标定好的相机外参是必需的,而且缺了会静默失败。** Context encoder 使用 PRoPE,它消费 `^{world}_{cam}T`,而且动作空间本身就定义在相机系下(`space_ee2cam`)。走 `joint7` 的话外参不进入动作链路,没有标定也能训 —— 但那时 `obs_extrinsics` 应当**整条轨迹全填单位阵**(退化成 base 系,自洽),绝不要填一半真一半假,并且要关掉 `shuffle_cameras`,否则两路几何上无法区分的相机被随机换序只是在加噪声。注意 PRoPE 和 RoPE 都是无参数的,所以切换 `pe_type` **不会**改变 `state_dict` —— checkpoint 依然会"加载成功",但位置编码的含义已经完全变了。症状:训练就是不收敛。录数据前先标定相机。

每一帧你需要有:

| 量 | 形状 | 说明 |
| --- | --- | --- |
| 末端位姿 | `(4, 4)` | `^{world}_{ee}T`,齐次矩阵。`ee_cam` 的监督目标;`joint7` 下仍需提供(用作 `cur_weT` 条件) |
| 关节角 | `(nq,)` | **弧度**。仅 `joint7` 需要,是它的监督目标 |
| 夹爪开合度 | 标量 | 归一化到 `[0, 1]`,0 = 闭合,1 = 张开 |
| 每路相机 RGB | `(H, W, 3)` | uint8 |
| 相机外参 | `(4, 4)` | `^{world}_{cam}T`,逐帧(腕部相机会动)。无标定时全填单位阵 |
| 相机内参 | `(3, 3)` | 针孔 `K`,每条 episode 内固定 |
| 时间戳 | 标量 | 秒,单调递增 |

世界系可以任意选取,但必须在**一条 episode 内保持一致** —— 模型只使用相对 camera 0 的位姿,原点不会泄漏进网络。

## 1. 把录制数据转成 HDF5

一条 episode 一个 `.h5` 文件,放在 `./data_converted/real_robot/` 下。参考实现见 `data_prepare/process_libero.py`。`DataSampler.sample_hdf5` 消费的布局如下:

```
episode_0001.h5
├── ee_pose            (T, 4, 4)  float32   ^{world}_{ee} T
├── gripper            (T,)       float32   [0 (闭合), 1 (张开)]
├── timestamp          (T,)       float32   秒
├── ee_pose_desired    (T, 4, 4)  float32   可选: 指令位姿(见下方说明)
├── gripper_desired    (T,)       float32   可选: 指令夹爪
├── exterior/                               <- 组名 = camera_names[0]
│   ├── rgb            (T, 3, H, W) uint8, 或 JPEG 字节的 vlen list
│   ├── pose           (T, 4, 4)  float32   ^{world}_{cam} T
│   └── K              (3, 3)     float32
└── wrist/                                  <- 组名 = camera_names[1]
    ├── rgb, pose, K   (同上)
└── .attrs["prompt_text"] = "pick up the red block"
```

说明:
- `ee_pose_desired` / `gripper_desired` 是可选的。如果存在,监督目标会用它们而不是**实际达到**的位姿 —— 这通常是真机上想要的行为:实际位姿滞后于指令,拿它训练等于教策略欠调。有条件的话请记录控制器 setpoint。
- `prompt_text` 是 HDF5 attribute,不是 dataset。名字里含 `prompt_text` 的多个 attribute 会被随机采样,这是加语言增强的廉价做法。单任务也需要至少一个,因为冻结的 SigLIP text encoder 总是会跑。
- 图像可以存原始数据或 JPEG 编码;`data_prepare/process_libero.py` 里的 `transform_image` 两条路径都有示例。

**训练前务必验证转换结果** —— 这能抓到外参和夹爪约定的错误,这类错误在 loss 上是完全看不出来的:

```bash
python datavis.py -l                # 列出已转换的数据集
python datavis.py RealRobot         # 叠加显示末端坐标轴和未来轨迹
```

如果叠加图里投影出的末端坐标轴没有落在真实夹爪上,说明外参或内参错了。先修这个,后面所有东西都不会work。

## 2. 声明数据集

编辑 `data_utils/datasets.py` 里 `RealRobot` 的四个 `TODO` 字段:

```python
class RealRobot(H5DatasetMapBase):
    config = DataConfig(
        sample_dt=1.0 / 15,                      # 1 / control_hz, 真实墙钟秒数
        record_dt=None,                          # None -> 从 `timestamp` 推断
        camera_names=("exterior", "wrist"),      # 第三人称相机放第一个
        ee_indices=(0,),                         # 单臂
        output_image_hw=(256, 256),
    )
```

`camera_names[0]` 不是无关紧要的顺序问题:它定义了整个动作表示的参考系(`main_cam_embed` 标记它的 token,`space_ee2cam` 把每个动作都表达在它的坐标系下)。把第三人称相机放第一个,与预训练数据保持一致(LIBERO 是 `agentview` 然后 `eye_in_hand`;DROID 是 `exterior` 然后 `wrist`)。

`sample_dt` 必须是真实的墙钟周期。它决定预测的 chunk 往前覆盖多久:`sample_dt * sample_state_gaps * num_future_states` 秒 —— 默认配置下是 2.13 秒。

## 2b. 如果数据是 memmap/bin 而不是 HDF5

上面的 h5 路径依赖 h5py 的惰性随机读:`H5DatasetMapBase.__getitem__` 每次只打开文件切几帧,`datavis.py` 和 `compute_action_stats.py` 也走同一条路。自己录的数据如果是「每条轨迹一个目录 + `metadata.json` 描述若干 `NAME.bin` 的 dtype/shape」,不需要转成 h5,也不需要改 `dataset_base.py` —— 需要对齐的只是 `__getitem__` 的**输出契约**:

| key | shape | 说明 |
| --- | --- | --- |
| `rgbs` | `(To, ncam, 3, H, W)` | **float32 且已 /255**,不是 uint8 |
| `K` | `(ncam, 3, 3)` | 必须对应 resize **之后**的 H/W |
| `obs_norm_xys` | `(To, ncam, 2, H, W)` | 由 `gen_norm_xy_map(H, W, K)` 生成 |
| `obs_extrinsics` | `(To, ncam, 4, 4)` | `^{world}_{cam}T` |
| `ee_poses` | `(nee, 4, 4)` | 只要最新一帧 |
| `history_actions` | `(nhist, nee, state_dim)` | `state_dim` 由动作空间决定 |
| `future_actions` | `(Ta, nee, state_dim)` | 同上 |
| `timestamps` | `(To,)` | |
| `valid_ee_mask` | `(nee,)` | bool |
| `prompt_text` | str | |

`data_utils/dataset_real.py` 里的 `RealBinDataset` 就是这样一个实现,直接改它的四个标注字段即可:`CAMERA_AXIS`(bin 里相机的轴顺序)、`IS_BGR`、`GRIPPER_MIN/MAX`(夹爪宽度到 `[0,1]` 的线性映射)、`ACTION_SPACE` / `NUM_JOINTS`。

它继承 `H5DatasetMapBase`(尽管完全覆盖了 `__getitem__`),因为 `concat_datasets` 要读 `cam_num` / `ee_num` 并回写 `pad2ncam` / `pad2nee`,`compute_action_stats` 要用 `skip_rgb`。基类的 `h5_filelist` 在这里存的是轨迹目录。

跑自带的契约校验:

```bash
python -m data_utils.dataset_real
```

它逐键校验形状,外加三个语义检查:`rgbs` 是否已除 255、夹爪是否在 `[0,1]`、以及按动作空间分支的量纲检查(EE 查位姿末行是否 `[0,0,0,1]`,关节查绝对值是否超过 2π——角度制/弧度制搞反在训练时完全不报错,只会让归一化范围大 57 倍)。

两点与 h5 路径的有意差异:memmap 路径**不做时间插值**(真机等间隔录制,直接按索引采样,`timestamps` 由 `index * record_dt` 合成);录制帧率不稳的话要改回按时间戳插值。以及无标定时 `obs_extrinsics` 全填 identity —— 这时 PRoPE 退化为 no-op、动作退化到 base 系,仍然自洽,但两路相机在几何上完全不可区分,所以必须 `shuffle_cameras=False`。

## 3. 微调

```bash
# ee_cam:从预训练权重出发
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config finetune_real \
  --pretrained_ckpt ./checkpoints/E2VLA/PRETRAIN_EXP_NAME/ckpt_xxxxxxx.pt \
  -s MY_ROBOT_EXP

# joint7:从头训
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config finetune_real_joint \
  -s MY_ROBOT_JOINT_EXP

# joint7:从 ee_cam 的预训练 ckpt 迁移主干(推荐,见 §3.7)
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config finetune_real_joint \
  --pretrained_ckpt ./checkpoints/E2VLA/PRETRAIN_EXP_NAME/ckpt_xxxxxxx.pt \
  --pretrained_strict False \
  --pretrained_ignore_action_layout True \
  -s MY_ROBOT_JOINT_EXP
```

`finetune_real_joint` 的默认值是按「从头训」配的:`lora_rank=0`、`max_lr=1e-4` 而不是 `5e-5`、`max_iterations=60k` 而不是 `20k`。走上面第二条命令(有预训练主干)的话,`max_iterations` 可以砍半,`lora_rank` 也可以调回 16 —— ContextEncoder 这时是预训练的,正好补上 LoRA 缺的那个前提。

两个 flag 缺一不可,而且不给的话会明确报错而不是静默降级:`--pretrained_ignore_action_layout True` 绕过 layout 戳,`--pretrained_strict False` 允许那 4 个张量形状不匹配。只给前者会直接抛错告诉你还差后者。

`finetune_real` 预设(`ee_cam`)与 LIBERO 那几个的差别都源于数据量更小:20k 步而不是 70k,`max_lr=5e-5` 而不是 `1e-4`,开启 `grad_clip=1.0`,`bs=16`,EMA 从第 1k 步开启。`sample_multiplex=1000` 把 episode 列表膨胀,让一个 "epoch" 有可用的长度。此外它默认 `lora_rank=16`,见下一节。

Checkpoint 加载行为:
- `--pretrained_ckpt` 默认**只加载权重**(`pretrained_weights_only=True`)。微调是一个新的优化问题:`current_iters` 归零、LR 从零重新 warmup,继承预训练的 Adam 动量会让最初几步的更新大于预期。传 `--pretrained_weights_only False` 恢复旧行为。
- 加载默认是 **strict** 的,并会准确报告哪些张量不匹配。如果看到 `context_encoder.proj_*` 或 `context_encoder.post_attn` 下的 shape mismatch,说明这个 checkpoint 早于 `compact model` 那次提交(122.4M vs 现在的 102.3M 参数)。要么换一个当前版本的 checkpoint,要么传 `--pretrained_strict False` 只加载兼容的那部分 —— 剩下的部分会从随机初始化开始训,而且这件事会在日志里大声说明,不会静默发生。
- `-c EXP_NAME`(续训)总是恢复完整的优化器状态,不受上述 flag 影响。

## 3.5 LoRA 微调(小数据集的默认方案)

在 `ContextEncoder` 上做 LoRA,DiffusionHead 保持全量训练。这是 `finetune_real` 的默认设置(`lora_rank=16`),理由很直接:ContextEncoder 占了 60% 的参数,而它学的是多视角几何融合 —— 这部分正好是从预训练里迁移得最好、也最不该被 70 条 episode 重新塑造的部分;DiffusionHead 才是需要拟合你这个任务动作分布的部分。

`vla_base` 上的实际参数量:

| 配置 | ContextEncoder 可训练 | DiffusionHead | 合计 |
| --- | ---: | ---: | ---: |
| `lora_rank=0`(全量,历史默认) | 60.95M | 41.39M | **102.34M** |
| `lora_rank=8` | 1.09M | 41.39M | 42.48M |
| **`lora_rank=16`(默认)** | **2.00M** | 41.39M | **43.39M** |
| `lora_rank=32` | 3.82M | 41.39M | 45.21M |

用法就是一个开关:

```bash
# LoRA (finetune_real 的默认行为,不需要额外传参)
python train.py --config finetune_real --pretrained_ckpt PRETRAIN.pt -s EXP

# 换 rank
python train.py --config finetune_real --lora_rank 8  --pretrained_ckpt PRETRAIN.pt -s EXP_R8

# 退回全量微调所有 102M 参数
python train.py --config finetune_real --lora_rank 0  --pretrained_ckpt PRETRAIN.pt -s EXP_DENSE
```

也可以给 LIBERO 预设加 `--lora_rank 16`,这四个 arm(0/8/16/32)天然构成一组消融。

### 具体训练了什么

LoRA 注入 `ContextEncoder` 中所有注意力投影(`to_q`/`to_k`/`to_v` 及其融合形式 `to_qkv` 等)。融合投影的秩按矩阵个数放大(`to_qkv` → `3r`),所以**每个单独的投影拿到的秩都是 `r`**。

除 LoRA 因子外,还保持可训练的有:**LayerNorm 的 affine 参数、所有 bias、`qformer.queries`、`main_cam_embed`**。这是标准的 "LoRA + norm/bias tuning" 配方 —— 它们几乎不花参数(合计 <1M),但能吸收低秩更新自己适应得很慢的分布偏移。其余部分(FFN、`proj_v`/`proj_l`/`proj_pe`)全部冻结。

注意 `replace_with_lora_linear` 本身只冻结它包裹的那些 Linear,剩下的是 `setup_lora` 显式冻的 —— 否则 FFN 那几十 M 参数会照常训练,那就不是 LoRA 了。

### 加载顺序是有约束的,不能颠倒

`LoraLinear` 会把 `...to_q.weight` 改名成 `...to_q.lin.weight`,所以 state_dict 的布局取决于有没有注入 LoRA:

| 场景 | 正确顺序 |
| --- | --- |
| 从预训练 ckpt 微调 | 先 `load_state_dict`(官方 ckpt 是纯净的,精确匹配),**再**注入 LoRA |
| 续训 `-c` | **先**注入 LoRA,再 `load_state_dict`(ckpt 里已经有 LoRA key) |
| 部署 | 注入 → 加载 → EMA `copy_to` → **最后** merge |

这些顺序在 `train.py` 和 `infer_utils/planner.py` 里已经写死了,正常使用不需要操心。需要知道的是:

- 顺序错了会抛 `RuntimeError` 并列出所有不匹配的 key,**不会静默地只加载一半**。
- Checkpoint 里存了 `lora_rank` 字段。续训时如果它和当前 config 不一致,会直接报错而不是加载失败。
- **`lora_rank > 0` 时必须传 `--pretrained_ckpt`**,否则直接拒绝运行:LoRA 冻结的基座权重如果本身是随机初始化的,等于 60% 的参数被永久卡在初始化状态。关节空间也满足得了这个前提 —— 用 §3.7 的跨空间迁移加载 `ee_cam` 的 ckpt,ContextEncoder 就是预训练的。`finetune_real_joint` 默认 `lora_rank=0` 只是因为它的默认配置是从头训。
- 部署时的 merge 必须在 EMA `copy_to` **之后**:EMA 的 shadow 参数就是 LoRA 因子本身,先 merge 会把 EMA 权重丢掉。

### 部署时的开销:零(LoRA)

`infer_utils/planner.py` 的 `load_model` 会自动读取 checkpoint 里的 `lora_rank`,注入、加载、然后调用 `merge_lora_linear` 把 `A @ B` 折进基座权重。merge 之后模型退回成一个普通的 `ActionExpert`,没有任何 `LoraLinear` 残留,推理没有额外算子和显存开销。整条链路(训练态输出 vs 部署态 merge 后输出)的数值漂移在 float32 下约 `1e-6`。

> 关于数值:LoRA 注入时 `B` 是零初始化的,所以 `lin(x) + (x @ A) @ B` 在函数意义上与 `lin(x)` 完全等价。但端到端测下来 context 输出会有约 `1e-6` 的漂移 —— 这不是逻辑错误:`lin(x) + 0` 会分配一个新张量,落在不同地址,可能走不同的 matmul/SDPA kernel 分块路径。已验证:把线程数钉到 1 时漂移显著减小,参数逐位相同,每个 `LoraLinear` 的输出与其基座 `Linear` 逐位相同。作为尺度参考,训练跑在 bfloat16 下,其 eps(~8e-3)比这个漂移大约 4000 倍。

### 视觉主干上的 LoRA(DINOv2 / SigLIP,默认关闭)

上面那套 LoRA 全部作用在 `ActionExpert` 内部,VLM 的两个主干始终是冻结的特征提取器 —— 这是本仓库的默认假设,所有已发布的 checkpoint 也都是这么训的。`vlm_lora_rank` 让你在**主干本身**上加 LoRA:

```python
# configs.py —— 只在预设里配,命令行不是它的入口
CONFIGS["finetune_real_joint_vlmlora"] = TrainConfig(
    ...,
    vlm_lora_rank=16,
    vlm_lora_targets=["dinov2", "siglip_vision"],  # 默认值,可加 "siglip_text"
)
```

什么时候值得开:预训练特征来自网络图片,而真机的光照、遮挡视野的夹爪、没有纹理的桌面,是 60M 的 ContextEncoder 只能绕开、没法修正的分布偏移。什么时候不该开:任何**先**该试的方案都还没试的时候 —— 这是整个流程里最贵的一个开关。

`vla_base` 上的实际参数量(rank 按每个投影计,DINOv2 的 `qkv` 是融合的所以拿 `3r`):

| `vlm_lora_rank` | DINOv2(86.58M) | SigLIP vision(92.93M) | 合计新增可训练 |
| --- | ---: | ---: | ---: |
| `0`(默认,全冻结) | 0 | 0 | **0** |
| `8` | 0.885M | 0.442M | 1.33M |
| `16` | 1.769M | 0.885M | **2.65M** |
| `32` | 3.539M | 1.769M | 5.31M |

参数量不是重点,**显存和步时才是**:一旦主干进入反传,两个 ViT 的全部激活都要为每个样本的每个相机保留下来。先把 `bs` 砍半再说。同时,任何"预计算/缓存冻结特征"的快路径都随之失效 —— 特征不再是图像的固定函数。

除 LoRA 因子外**什么都不训**:没有 bias/norm tuning。这与 ContextEncoder 那套配方相反,理由也直接 —— 主干的 norm 统计量正是最该原样保留的部分,而且这里没有"在本机器上预训练过"的状态可供重新拟合。

关于目标选择:`siglip_text` 可选但默认不开。单任务微调集只有一条 prompt,适配文本编码器等于去拟合那一个字符串。

三个必须知道的行为:

- **因子会随 checkpoint 走,但主干权重不会。** 主干是构建时从 HuggingFace / torch.hub 拉的,从来不进 state_dict。所以 `vlm_lora_weights` 是"这套权重对应的特征不是原版特征"的**唯一**记录。一个开了 VLM LoRA 训出来的 checkpoint,如果在评测时没重新注入,会**一个 key 都不缺地加载成功**,然后让 action expert 去读它从没见过的特征。`check_vlm_lora` 就是为了让这种不匹配变响。
- **部署时不 merge**,与 ActionExpert 相反。主干是 bf16,而 `round_bf16(W + AB)` 无论 `AB` 多小都会在 `W` 上引入约 `3.3e-3` 的相对误差。实测(768 宽的 bf16 Linear):一个对输出影响为 `2.3e-3` 的 LoRA,merge 会再叠上 `2.7e-3` 的误差 —— 适配量整个埋进舍入噪声;再小十倍时只有 36% 的 `AB` 能从舍入里活下来。所以推理保留 `LoraLinear`,代价是每个注意力投影多一次 rank-r 矩阵乘(相对 ViT 前向可以忽略),换来评测与训练算的是同一个函数。
- **主干里所有 `no_grad` 都变成了条件式的**(`models/layers/utils.py:maybe_no_grad`)。没开 LoRA 时行为逐位不变,走的还是原来的 `no_grad`;开了之后只有被适配的那个塔解除 `no_grad`。这一步不是可选的:原来那个无条件的 `no_grad` 会把 LoRA 因子静默地从计算图里摘掉 —— 参数照样在优化器里、照样进 checkpoint,只是永远不动,而且不报任何错。

## 3.5b 可训练的卷积旁路(`conv_tower`,默认关闭)

`vlm_lora_rank` 和它解决的是同一个问题 —— 预训练特征来自网络图片,真机的光照、遮挡视野的夹爪、没有纹理的桌面在那个分布之外 —— 但代价完全不同。`vlm_lora_rank` 要为两个 ViT 的每个相机、每个样本保留全部激活做反传;`conv_tower` 只是在旁边加一条 **2.9M 参数、完全可训练**的浅层 CNN,两个主干仍然一动不动地冻着。

结构在 `models/conv_tower.py`:ResNet18 **只取到 `layer2`**(`layer3`/`layer4` 根本不构造),直接吃数据集出来的 RGB 原图,不做任何 resize。`256×256` 输入经 `conv1 → maxpool → layer1 → layer2`(共 /8)再过一个 stride-2 的 head conv,落到 `16×16 = 256` token —— 与两个 ViT 的 patch 数一致。留前两段而不是整个网络,是因为浅层带的是边缘、纹理、局部颜色统计,恰好是 patch-16 的 ViT 丢掉的、也恰好是几百条示教就能学出来的;深层语义那一端冻结的 ViT 本来就做得好。

```bash
# 两个现成预设,都需要 --pretrained_ckpt
python train.py --config finetune_real_conv \
  --pretrained_ckpt ./checkpoints/E2VLA/PRETRAIN/ckpt_xxxxxxx.pt -s FT_CONV
python train.py --config finetune_real_conv_lora \
  --pretrained_ckpt ./checkpoints/E2VLA/PRETRAIN/ckpt_xxxxxxx.pt -s FT_CONV_LORA
```

| 预设 | `lora_rank` | `conv_tower` | 说明 |
| --- | ---: | --- | --- |
| `finetune_real` | 16 | `none` | 基线 |
| `finetune_real_conv` | 0 | `resnet18` | 主干全开 + 旁路 |
| `finetune_real_conv_lora` | 16 | `resnet18` | LoRA 主干 + 全量训练旁路 |

三个都用同一个预训练 checkpoint 跑,才是可比的对照。

### 融合方式,以及为什么预训练权重仍然精确可用

`ContextEncoder` 原来是 `proj_v[0](dinov2) + proj_v[1](siglip)`。加旁路后变成:

```python
x_sem = proj_v[0](dinov2) + proj_v[1](siglip)     # 一个字没改
x_cnn = proj_v[2](conv_tower(rgb))                 # 新增,追加为 index 2
x_v   = proj_fuse(cat([x_sem, x_cnn], dim=-1))     # 新增
```

两点是刻意的:

- **dino+siglip 的相加保持原样**,新模态从旁边 concat 进来。那个相加关系正是预训练主干学到的东西。追加成 `proj_v[2]` 也意味着 `proj_v.0.*` / `proj_v.1.*` 的 key 一个都没变。
- **`proj_fuse` 初始化成 `[I | 0]`**,所以 `x_v ≡ x_sem`,第 0 步与不带旁路的模型**逐位相同**(实测 `max|Δ| = 0`)。这与 `proj_pe` 的零初始化、LoRA 的 `B` 零初始化是同一个手法。concat 之后再融合严格比相加更有表达力 —— 相加只是 `[I | I]` 这一个特例。

因此从官方 checkpoint 微调时**不需要** `--pretrained_strict False`。`load_actor_weights` 新增了 `allow_missing_prefixes`,`train.py` 按 `CONV_BRANCH_KEYS` 填进去:只有这三个新模块允许缺失,其它任何 missing / unexpected / 形状不符照样抛。日志会明确列出来:

```
[INFO] pretrained checkpoint: 146 tensors matched. 74 tensors belong to modules this run
       adds and the checkpoint cannot contain; they keep their initialisation and are trained:
           context_encoder.conv_tower.* (66 tensors)
           context_encoder.proj_v.2.* (6 tensors)
           context_encoder.proj_fuse.* (2 tensors)
```

### 学习率单开一组

主干是热启动的(想小步走),旁路是 ImageNet 初始化后要适配新域的(想大步走),共用一个 `max_lr` 两头不讨好。`conv_tower_lr_scale` 给旁路单独一组 `lr = max_lr * scale`(两个预设都是 `2.0`,是起点不是调好的值)。scheduler 是 `LambdaLR`,按各组自己的 base_lr 等比缩放,所以这个比例在整个 warmup/constant 过程中保持不变。

注意 `proj_fuse` **不在**这一组里 —— 它是 `[I | 0]` 起步的,用旁路那个更大的学习率会在头几步就把精确热启动撕掉,它属于主干的节奏。

### 三个必须知道的行为

- **和 LoRA 共存是成立的,而且是预期用法。** `setup_lora` 会把 `ContextEncoder` 里除因子和白名单外的一切冻结,所以 `CONV_BRANCH_KEYS` 被加进了 `LORA_KEEP_TRAINABLE`。不加的话旁路会**静默地冻在 ImageNet 初始权重上**,而所有日志仍然把它算作模型的一部分。新加的模块没有可供低秩分解的预训练基底,只能全量训。
- **`conv_tower` 必须留在 `model_kwargs()` 里。** 旁路的张量是**进** checkpoint 的,而 `infer_utils/planner.py` 只能从 dump 出来的 config json 知道要不要重建它。漏了会在 `load_state_dict` 处报错 —— 报得很响,但要等到主干和仿真器都起来之后。
- **没有新增 checkpoint 戳,这是刻意的。** 其它四个戳存在的理由是"张量名和形状完全相同、只有语义不同";旁路带来的是**新 key**,`CkptCompatReport` 会逐条列名、`planner.py` 的严格 `load_state_dict` 会直接拒绝,再加一个戳是冗余。

### 代价

参数 +2.9M(相对 102.3M 可训练参数约 +2.8%),但**真正的代价是步时和显存**:`layer1` 在 `64×64` 分辨率上跑。按 FLOPs 估算(未实测)前向加反传会让 step time 涨到 1.5~2 倍,激活显存同步上升 —— 第一次跑请自己记一下这两个数。跑不下就先砍 `bs`。

另外一个已知且刻意不修的性质:本卷积栈的 token `p` 中心落在输入像素 `16p`,而 ViT 的 patch `p` 覆盖 `[16p, 16p+16)`、中心在 `16p+7.5` —— 半个格子的偏移,ResNet stem 换任何 padding 都消不掉。`proj_pe` 和后面的注意力足以吸收半个 patch 的平移,而且这条支路本来就是训出来的。

## 3.6 动作归一化(q01/q99)

默认**关闭**,`action_norm_stats` 不设就是历史行为。开启后,模型动作空间的每个通道按其 1%/99% 分位数线性映射到 `[-1, 1]`。

### 为什么

模型的动作不是数据集里给的原始状态。`ee_cam` 下它是 `space_ee2cam` 产出的**相机系相对量**:3 维平移 + 6D 旋转 + 夹爪开合,共 10 维;`joint7` 下是绝对关节角 + 夹爪,共 8 维。下面以 `ee_cam` 为例——这 10 个通道的量纲差得很远 —— 平移是米(约 `1e-2`),而一个接近单位阵的 delta 的 6D 旋转是 `[1, 0, 0, 0, 1, 0]`,两个通道钉在 1 附近、四个钉在 0 附近。DDIM 从 `N(0, I)` 采样并预测 epsilon,隐含假设被去噪的干净数据是单位尺度的,现状并不满足。

用分位数而不是均值/方差:遥操作数据里有罕见的大跳变(跟踪丢失、操作员复位),标准差会被这些尾巴主导,反而把真正重要的那 98% 的运动压扁。

### 用法

```bash
# 1. 统计。必须对着将要训练的那个 config 算 —— 统计量是 config 的属性,不是磁盘上数据的属性
#    (它读 config 的 action_space,在对应空间里测量;此时 action_norm_stats 应仍为 None)
python -m data_prepare.compute_action_stats --config finetune_libero_10 \
  -o ./action_stats/libero_10.json

# 关节空间同理
python -m data_prepare.compute_action_stats --config finetune_real_joint \
  -o ./action_stats/real_joint7.json

# 2. 训练时指向它
CUDA_VISIBLE_DEVICES=0 python train.py --config finetune_libero_10 \
  --action-norm-stats ./action_stats/libero_10.json \
  --pretrained_ckpt PRETRAIN.pt -s FT_EXP

# 3. 评测不需要任何额外参数,统计量已经存进 checkpoint 了
```

统计脚本不解码图像(`skip_rgb`),因为动作空间不依赖像素,而解码几乎是全部的耗时;除像素外的采样、对齐、时间窗口都与训练完全一致。它默认同时统计 `history_actions` 和 `future_actions`(两者共用同一个 normalizer),`--future-only` 可以只统计预测的 chunk。

### 几个不显然但重要的点

- **统计量不可跨数据集复用。** 它是在相机系相对动作上算的,依赖 DataConfig(哪几个相机、`sample_state_gaps`、未来时域、是否 shuffle 相机)。换一个微调集就重算一次。
- **统计量按动作空间打了 layout 戳,不可跨空间复用。** json 里的 `"layout"` 字段(`cam_rel_t3r6_openness` 或 `abs_joint7_openness`)在加载时严格比对。这不是多余的谨慎:两个空间的通道数可能撞车(9 轴机械臂的关节空间也是 10 维),那时每个张量的名字和形状都对得上,加载报告「干净匹配」,然后模型把关节角当成米和旋转去去噪。没有 `"layout"` 键的旧文件一律按 `ee_cam` 读(它们都是)。
- **clip 不作用于旋转通道**(`DEFAULT_CLIP_DIMS = (0, 1, 2, 9)`),这是 `ee_cam` 专有的;关节空间每一维都是度量量,全部参与 clip。平移和开合度是度量量,超界意味着危险的跳变,该 clip;6D 旋转要经过 `rotation_6d_to_matrix` 的 Gram-Schmidt,是「看方向、不看模长」的量,超界无害,而 clip 会把整个越界半空间塌缩到同一个角点 —— 一旦两个 3 维向量共线,Gram-Schmidt 会返回一个带零行的矩阵,那根本不是旋转矩阵。未训练的模型第一次前向就能走到这个状态。
- **近似常数的通道会被自动跳过**(`q99 - q01 < 1e-6` 时 scale=1、offset=0),不会除以 0。脚本会在表格里把这些通道标出来 —— 看到它先怀疑是不是采样窗口太少,而不是数据真的退化。
- **归一化不改变任何张量的名字和形状。** 一个在归一化动作上训练的 checkpoint,加载到没有 normalizer 的模型里会**零缺失 key 地加载成功**,loss 曲线看着正常,然后机器人执行的动作差了大约一个仿射的倒数。因此统计量会被写进 checkpoint 的 `action_norm` 字段,`train_utils/ckpt.py:check_action_norm` 在续训时严格比对(不一致直接报错),从预训练 ckpt 微调时降级为大声警告(拿官方无归一化的 ckpt 配上自己数据的分位数微调是合理操作,但不该是无意中发生的)。这和 `objective` 那个 stamp 防的是同一类问题。
- 推理侧的解析顺序和 `lora_rank` 一致:**checkpoint 里的记录优先于 config json**,因为同一个目录下的 json 可能被后来的 run 覆盖掉。

## 3.7 动作空间:关节角 vs 末端位姿

`TrainConfig.action_space` 选择模型在哪个空间里去噪,实现在 `models/action_space.py`。两个空间是并存的策略对象,`ee_cam` 的行为与引入这个开关之前逐行等价。

| | `ee_cam` | `joint7` |
| --- | --- | --- |
| `state_dim`(数据集给的) | 17 = 展平 4x4 + 夹爪 | nq+1 = 8 |
| `action_dim`(模型内部) | 10 = 3 平移 + 6D 旋转 + 夹爪 | 8 |
| `layout` 戳 | `cam_rel_t3r6_openness` | `abs_joint7_openness` |
| loss 切分 | pos / rot / openness | joint / openness |

注意 `state_dim` 和 `action_dim` 是两个数。在 `ee_cam` 下它们分别是 17 和 10,`space_ee2cam` 是中间的转换;`joint7` 下两者相等,但代码里仍然分开,不要混用。

### 为什么关节角用绝对值而不是增量

这与 `ee_cam` 的取向相反,是有意的。EE 那边预测增量,是因为绝对世界位姿依赖标定、跨 episode 不可比;关节角本身就在机器人自己的坐标系里,绝对值天然可比,而增量会把误差逐步累积到轨迹末端。ACT / Diffusion Policy 这一系的关节空间实现也都是绝对目标。

要改成增量的话,覆盖 `AbsJoint` 的 `states2action` / `action2states` 即可,但**必须同时改 `layout` 字符串**,否则旧的统计文件会静默套用到新编码上。

### 预训练权重仍然能用,只是不是精确加载

和动作编码绑定的只有三层:`dp_head.hist_enc.0`、`dp_head.traj_enc.0`、`dp_head.act_head.3`,合计约 2.2 万参数。`ContextEncoder` 的 60.95M 和头部整个 DiT 栈都与动作空间无关,形状分毫不变。实测把 `ee_cam` 的 checkpoint 加载进 `joint7` 模型:

```
transferable: 101.73M / 101.74M = 99.98%
shape mismatch: 4 个张量
    dp_head.hist_enc.0.weight   (768, 9)  vs (768, 7)
    dp_head.traj_enc.0.weight   (768, 10) vs (768, 8)
    dp_head.act_head.3.weight   (10, 768) vs (8, 768)
    dp_head.act_head.3.bias     (10,)     vs (8,)
unexpected: abs_pos_enc 的 6 个张量(关节空间不建它)
```

用 `--pretrained_strict False --pretrained_ignore_action_layout True` 开启。两个 flag 分工不同:前者允许形状不匹配的部分加载(仓库本来就有的机制),后者显式绕过 layout 戳。默认都关着,因为跨空间**误**加载一旦发生是查不出来的;要绕过就必须是有意的。

有了预训练主干,**LoRA 也重新可用了** —— `lora_rank > 0` 缺的那个前提(基座权重不能是随机初始化的)正好被补上。

不要试图用「保留 10 维输出、只监督前 8 维」的办法来强行精确加载。形状能对上,语义对不上:通道 0 上的预训练权重编码的是「相机系平移的米数」,把关节角 q0 喂进去不是迁移;夹爪会落到 r11 上,而 6D 旋转那 6 维是**刻意排除在 clip 之外**的(`DEFAULT_CLIP_DIMS`),夹爪坐那儿会失去越界保护。更硬的障碍是 `pos_rel2abs` 读 `[..., :9]` 做 SE(3) 代数,关节角坐在那里会算出垃圾位置并加进特征 —— 要么关掉这条路(那 `abs_pos_enc` 本来就是死权重),要么它每一步都在污染特征。openpi 的 padding 是为了在**一个模型里 batch 不同 DOF 的机器人**,被 pad 的维度都是同一种量,那是另一个问题。

### 关节空间放弃了什么

1. **头部的绝对位置编码。** `DiffusionHead.pos_rel2abs` 把预测的 t3r6 还原成相机系下的末端绝对位置,再作为一路附加位置编码加回特征。关节角要做这件事需要正运动学(得有 URDF/DH 参数),所以 `joint7` 下 `abs_pos_enc` 根本不建 —— 不是建了不用,是不进 `state_dict`,免得每个 checkpoint 里躺着一堆拿不到梯度的死权重。有 DH 参数的话,在 `AbsJoint` 里实现 FK 并把 `has_pose_geometry` 打开就能补回来。
换来的是:模型输出直接就是关节控制器的输入,不过逆运动学,手眼标定误差也不再进入动作链路。

### 防串用的两道戳

`layout` 会同时写进统计 json 和 checkpoint(`action_layout` 字段),在 `build_action_normalizer` 和 `train_utils/ckpt.py:check_action_layout` 两处校验。理由和 `objective` 那个戳完全一样(见 §3.8):两个空间的通道数可能撞车(9 轴机械臂的关节空间也是 10 维),那时**每个张量的名字和形状都对得上**,权重零缺失 key 地加载、布局检查报告干净匹配,然后模型开始输出垃圾。名字和形状分辨不了,只有这个戳能。

早于这两个戳的文件一律按 `ee_cam` 读 —— 它们确实都是。

## 3.8 生成目标:DDIM 还是 Flow Matching

动作头支持两种生成目标,由 `TrainConfig.objective` 选择:

| | `"ddim"`(默认) | `"flow"` |
| --- | --- | --- |
| 前向过程 | 余弦噪声表,`diffusion_timesteps=100` 步 | 直线 `x_t = (1-t)·noise + t·action`,`t ∈ [0,1]` |
| 网络预测 | 混进去的噪声 epsilon | 路径上的速度 `action - noise` |
| 采样 | DDIM 反向循环,默认 20 步 | 显式 Euler 积分,默认 10 步 |
| 预训练 ckpt | 官方发布的都是这个 | 得自己从头 pretrain |

**网络本身一个字都没改。** `DiffusionHead` 始终是 `(时间, x_t, 上下文) -> action_dim 的向量`;变的只有三件事:训练时 `x_t` 怎么造、输出拿什么去回归、采样时怎么把输出积分回动作。想换 objective 就改这一个字段:

```bash
# 用预设(它们只是把对应的 DDIM 预设的 objective 改成 flow)
CUDA_VISIBLE_DEVICES=0 python train.py --config finetune_libero_10_flow -s EXP
CUDA_VISIBLE_DEVICES=0 python train.py --config pretrain_flow -s EXP

# 或者在任意预设上直接覆盖(tyro 接管 --config 之后的所有参数)
CUDA_VISIBLE_DEVICES=0 python train.py --config finetune_real --objective flow -s EXP
```

评测端不需要任何额外参数:`objective`、采样步数这些都会跟着 config json 存进 checkpoint 目录,`infer_utils/planner.py` 从那里重建模型。

### 换它图什么

**推理步数**。同等质量下 flow 通常只要 DDIM 一半的网络前向 —— 动作头每一步都要跑一遍 DiT,这在实机控制频率上是直接的开销。OT 路径上噪声和动作之间的真实轨迹是一条匀速直线,离散化本身不产生误差,所以均匀步长、无调度器的 Euler 就够了,`inference_timesteps` 就是精确的前向次数。

代价是 flow 的 checkpoint 和 DDIM 的**不能互相加载**(`check_objective` 会拦)。但**主干是能迁移的** —— 见下面"用 DDIM 预训练权重初始化 flow 运行"。

### 一个不显然的实现细节:时间要缩放,而且要翻向

`ActionExpert.head_time` 喂给头部的不是 `t`,是 `(1 - t) × diffusion_timesteps`。两个理由,缺一不可。

**缩放。** 头部的时间嵌入是 `SinusoidalPosEmb(temperature=1e4)`,当初是照着 `[0, diffusion_timesteps)` 上的整数步设计的。直接把 `t ∈ [0,1]` 喂进去,所有样本会挤在这个范围最前面的 1% 里,除了最高频的几个频带以外全是平的 —— 头部近似于对时间失明,学出来的是一个平均速度场。这个失败**是安静的**:loss 照样下降(平均场确实是一个极小点),只是采样出来的动作糊向均值。

**翻向。** 这类嵌入索引的是**噪声水平**,而两种目标的编号方向正好相反:DDIM 的 timestep 0 是干净数据、`diffusion_timesteps` 是纯噪声;flow 的 `t=0` 是纯噪声、`t=1` 是数据。喂 `1 - t` 之后,这个参数在两种目标下都表示"噪声占比"(也正是 pi0 喂的东西,它的 t 本来就是噪声占比)。不翻的话,DDIM 预训练的头部一上来对"我的输入有多脏"的判断是**完全反的**。

两者都是 `[0,1]` 上的双射,所以从头训 flow 对这两个选择无所谓;只有迁移和可读性在乎。

### 用 DDIM 预训练权重初始化 flow 运行

**可以,而且是推荐做法。** 动作专家 102.3M 参数里有 96.5% 跟 objective 无关:

| 模块 | 参数量 | 占比 | 跨 objective |
| --- | --- | --- | --- |
| `context_encoder` | 60.95M | 59.6% | 原样可用 |
| `dp_head.traj_context_attn`(头部 DiT) | 37.81M | 37.0% | 原样可用 |
| `hist_enc` / `traj_enc` / `abs_pos_enc` / 两个时间嵌入 | 2.98M | 2.9% | 原样可用 |
| `act_head` 的输出 Linear | 7.7k | 0.008% | **重置** |

道理是:除了输出层,头部里所有东西都在**编码**输入 —— 动作块、历史、时间、位置 —— 而这些输入在两种目标下活在同一个空间里。只有读出层的含义变了:epsilon 头预测噪声,flow 头预测 `action - noise`,在高噪声端后者几乎就是前者的**相反数**。把训好的输出层搬过去,等于让新 run 从一个系统性反号的输出开始 —— 比从零开始更差,而且表现为"收敛慢"而不是报错。所以 `train.py` 在迁移时把它清零,回到 from-scratch init 的状态,喂给它的那一层保留。

```bash
python train.py --config finetune_libero_10_flow \
  --pretrained_ckpt PRETRAIN_DDIM.pt \
  --pretrained_ignore_objective -s EXP
```

几点:
- **不需要**放宽 `pretrained_strict`。每个张量都名字形状对得上 —— 这恰恰就是危险所在,所以才要一个显式 flag。
- 只对 `--pretrained_ckpt` 生效。`-c` 续训永远要求精确匹配:续训是继续同一个优化问题。
- 和 `pretrained_ignore_action_layout` 可以叠加(跨 objective + 跨动作空间),那时 `hist_enc.0` / `traj_enc.0` / `act_head.3` 因为形状不同也会重置,还要再加 `--no-pretrained_strict`。
- tyro 把 bool 字段渲染成裸 flag,不是 `--flag True`:开是 `--pretrained_ignore_objective`,关是 `--no-pretrained_ignore_objective`(实测 tyro 1.0.15)。

### 训练时的 t 怎么采

`flow_time_sampling` 三选一,只在 `objective="flow"` 时读:

- `"uniform"` —— rectified flow 原论文的做法,没有旋钮。**默认。**
- `"logitnormal"` —— `sigmoid(N(0,1))`,SD3 的做法,权重压在路径中段(两端接近平凡:t→0 时答案几乎就是 `-noise`,t→1 时几乎是残差)。
- `"beta"` —— `1 - Beta(flow_time_alpha, 1)`,pi0 的做法换算到本仓库的 t 约定(pi0 的 t 方向是反的)。质量压在高噪声端,那里的误差会被之后每一个 Euler 步继承,代价最大。

少步数采样看着欠收敛时,先试后两个。

### 防串用的第三道戳

`objective` 的戳和 §3.7 那两道是同一类东西,而且是其中最危险的一个:两种目标**共用每一个张量的名字和形状**,一个 flow 的 checkpoint 加载进 DDIM 模型会零缺失 key 地成功、布局检查报告干净匹配,然后把一个速度场喂给噪声预测的采样器。所以 `train_utils/ckpt.py:check_objective` 在不匹配时**直接报错**,不降级成警告 —— 和动作归一化那道戳不同,这里没有"警告一下继续"的中间地带,要么是意外(必须拦死),要么是上面那种明确的主干迁移(走 `pretrained_ignore_objective`,而且会顺手重置输出层)。

不带 `objective` 字段的 checkpoint(官方发布的预训练权重)一律按 `"ddim"` 读 —— 它们确实都是。

### 两种 objective 的 loss 数值不可比

`action_space.loss` 的 30/10/10 权重在两种目标下性质不同:epsilon 目标的每一段都是同一个标准正态的切片,权重纯粹是 loss 整形;flow 的目标 `action - noise` 带着动作空间自己的量纲,同一组权重就顺带成了(粗糙的)量纲补偿。把两条 loss 曲线并排看没有意义,判断 flow 跑得好不好要看 rollout —— 或者看 §3.9 的 `sample/*`,那组数是跨 objective 可比的。

## 3.9 训练监控(TensorBoard)

```bash
tensorboard --logdir ./logs/E2VLA
```

命令行这边是一条 tqdm 进度条(整个 run 一条,按 `max_iterations` 计,不是每个 epoch 一条;`-c` 续训会从 checkpoint 的迭代数接着走),后缀实时显示各 loss 分项和 lr。进度条画在 **stderr**,循环里其它输出走 `tqdm.write` 到 **stdout**:NaN 跳过、`Save to …`、`sample/*`,以及每 `log_interval` 一行完整的 `[INFO] iter/max | 各项 loss`(和 TensorBoard 写点、平均值清零是同一个边界)。所以 `python train.py > train.log` 得到的是干净的低频日志,进度条留在终端上。



写出的标签按前缀分组,除 `sample/`、`data/rgb`、直方图外都是默认开启、且开销可忽略的:

| 分组 | 内容 | 它回答的问题 |
| --- | --- | --- |
| `train/` | `total_loss` 和各分项(`pos_loss` / `rot_loss` / `joint_loss` / `openness_loss`) | 原有行为,标签没变 |
| `diag/` | `cos_sim`、`pred_std` / `target_std`、`loss_noise_b0..3`、`act_absmax` / `act_clip_frac` | loss 看着正常但模型其实坏了的那几种情况,见下 |
| `optim/` | `lr`、`grad_norm`、`clip_frac`、`param_norm`、`update_norm`、`update_ratio`、`grad_scale` | 学习率选得对不对;梯度裁剪是不是一直在生效 |
| `gnorm/` | 按模块分的梯度范数:`context_encoder` / `dp_head` / `vlm_lora` / `conv_tower` / `total` | "loss 不动了"到底是哪一半不动了 |
| `perf/` | `it_per_sec`、`samples_per_sec`、`data_wait_ms`、`step_ms`、`gpu_mem_gb`、`nan_skip_frac` | 是不是在等 dataloader(那要调 `workers`,和模型无关) |
| `progress/` | `epoch`、`samples_seen` | 迭代数之外的进度 |
| `data/` | `valid_ee`,以及可选的输入图像与 prompt | 喂进去的到底是什么 |
| `sample/` | `pos_err_m` / `rot_err_deg` / `grip_acc`(关节空间是 `joint_err_rad` 等) | **默认关闭**,见下 |
| TEXT 页 | 解析后的完整 config、可训练参数表、这次运行的来源(预训练 ckpt / resume / 数据集) | 三个月后回来对比两次运行时唯一需要的东西 |

三个可选开关(都在 `TrainConfig` 里,命令行可以直接覆盖):

```bash
# 每 500 步跑一次完整采样,按物理单位报告动作误差
python train.py --config finetune_real -s EXP --log_sample_interval 500
# 前期确认输入没问题:每 1000 步存一张各相机的输入图 + prompt
python train.py --config finetune_real -s EXP --log_image_interval 1000
# 逐张量的权重/梯度直方图,只在怀疑某层饱和或死掉时开
python train.py --config finetune_real -s EXP --log_hist_interval 5000
```

### `sample/*`:唯一和 rollout 同量纲的训练期指标

训练 loss 衡量的是**单步去噪**回归得准不准,它可以一直降,而积分出来的整条轨迹并没有变好;两种 objective 的 loss 更是彼此不可比(§3.8)。`log_sample_interval > 0` 时会在当前 batch 上跑完整的采样循环,把结果解码回世界坐标系,报告米 / 度 / 夹爪准确率 —— 和评测同量纲,而且跨 objective、跨 `action_norm` 配置都可比。代价是一次额外前向加 `inference_timesteps` 次 head 调用,所以按间隔跑,500~1000 基本不影响吞吐。

两点提醒:它测的是训练 batch(本仓库没有验证集划分),所以读作"采样器能不能复现它训过的东西",泛化仍然要跑 `examples/libero/eval.py`;开了 EMA 时它测的是在线权重,不是部署用的 EMA 影子。

### `diag/*`:loss 正常但模型不对的三种情况

* **塌成均值。** head 如果无视时间条件,只学一个平均场,loss 照样降(平均场确实是个极小点),采样出来的动作则糊向均值。表现是 `diag/pred_std` 逐渐偏离 `diag/target_std`,以及尺度无关(因而不受 loss 权重影响)的 `diag/cos_sim` 卡在 0 附近不涨。
* **路径的一端没训到。** `diag/loss_noise_b0..3` 把**未加权**的 L1 按输入被污染的程度分了四桶,b0 接近干净、b3 接近纯噪声。高噪声桶一直降不下来,意味着采样的每一步都在继承这个误差 —— `flow_time_sampling="beta"` 就是为这种情况准备的(§3.8)。
* **归一化饱和。** `diag/act_clip_frac` 是目标落在 [-1, 1] 之外的比例。配了 `action_norm_stats` 时,那正好是落在 q01/q99 之外的动作:百分之几是正常的,比例很大说明统计文件来自别的数据集或别的 config(§3.6),而这件事没有任何别的地方会报出来。

## 4. 部署

> **关节空间目前只支持到训练,部署链路还没改。** `TrajPlanner` 的动作解码是 17 维 SE(3) 专用的:`_make_empty_action` 写死 `16+1` 并填单位阵,`get_action` 做 `reshape(actions[..., :16], (..., 4, 4))` 后返回 `future_ee_poses`,`ensemble_traj` 也是按位置 + 旋转分别插值的。用 `joint7` 的 checkpoint 起 `remote_service`,模型本身会正确构建(`load_model` 已经读 config 的 `action_space` 并校验 checkpoint 的 layout 戳),但 `get_action` 会在 reshape 处炸掉 —— 8 维状态没法 reshape 成 4x4。要补的是:`_make_empty_action` 按 `state_dim` 分配、`get_action` 按动作空间分支返回关节目标而不是位姿、以及一个关节空间的 ensembler(关节角可以直接线性平均,比 SE(3) 那套简单)。
>
> 下面这节描述的是 `ee_cam` 的部署流程。

和 LIBERO 一样的三进程结构,只是客户端喂的是真实观测而不是仿真器。启动命名服务和策略服务:

```bash
pyro4-ns
CUDA_VISIBLE_DEVICES=0 python -m infer_utils.remote_service \
  --ckpt ./checkpoints/E2VLA/MY_ROBOT_EXP/ckpt_xxxxxxx.pt \
  --uri MY_ROBOT --ensemble 3
```

只有在训练确实开了 EMA 时才加 `--ema`(`finetune_real` 是开的)。LoRA 不需要额外参数,`lora_rank` 和 `vlm_lora_rank` / `vlm_lora_targets` 都从 checkpoint 里读:前者注入后 merge,后者注入后**保持不 merge**(理由见 §3.5)。

你的控制循环参照 `examples/libero/eval.py`:

```python
controller = get_shm_proxy("MY_ROBOT", ns_host="localhost", ns_port=9090)
controller.set_config("RealRobot")     # 按数据集类名索引;bin 数据集是 "RealBinDataset"
controller.reset()                     # 每条 episode 之间必须调用
controller.set_prompt("pick up the red block")

while not done:
    controller.add_obs_frame(obs_frame)          # 见 TrajPlanner.add_obs_frame
    future_ee_poses, future_grippers, future_time, _ = controller.get_action()
    future_ee_poses  = future_ee_poses[:, 0]     # ee_indices == (0,)
    future_grippers  = future_grippers[:, 0]
    future_ee_poses, future_grippers = controller.ensemble_traj(
        future_ee_poses, future_grippers, future_time)
    # 执行前几个 waypoint, 然后重新规划
```

episode 之间的 `controller.reset()` 不是可选的:ensembler 是按时间戳索引的,而新 episode 的时间戳会重新从 0 开始,不 reset 的话上一条 episode 的残留 chunk 会被混进新 episode 的最初几个动作里。

`obs_frame` 使用与 HDF5 转换相同的 schema —— 精确的 dict 布局见 `TrajPlanner.add_obs_frame` 的 docstring,完整示例见 `examples/libero/eval.py` 里的 `obs_libero2ours`。
