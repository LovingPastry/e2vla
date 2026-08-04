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
日志保存到 `./logs/E2VLA/EXPERIMENT_NAME`,checkpoint 保存到 `./checkpoints/E2VLA/EXPERIMENT_NAME`。

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

`layout` 会同时写进统计 json 和 checkpoint(`action_layout` 字段),在 `build_action_normalizer` 和 `train_utils/ckpt.py:check_action_layout` 两处校验。理由和 `objective = "ddim"` 那个戳完全一样:两个空间的通道数可能撞车(9 轴机械臂的关节空间也是 10 维),那时**每个张量的名字和形状都对得上**,权重零缺失 key 地加载、布局检查报告干净匹配,然后模型开始输出垃圾。名字和形状分辨不了,只有这个戳能。

早于这两个戳的文件一律按 `ee_cam` 读 —— 它们确实都是。

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

只有在训练确实开了 EMA 时才加 `--ema`(`finetune_real` 是开的)。LoRA 不需要额外参数,`lora_rank` 从 checkpoint 里读。

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
