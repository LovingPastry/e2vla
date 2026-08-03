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

**推荐从预训练 checkpoint 出发,但这不是硬性要求** —— 在 50-100 条演示上做单任务行为克隆是标准做法(Diffusion Policy、ACT 都是这个量级的模型和数据)。把它当成一个实验问题:如果能跑两次 20k 步,直接测一下。

预训练 checkpoint 具体带来什么:可训练参数的 60% 在 `ContextEncoder` 里,它的职责是通过 PRoPE 融合多路相机视角,再经 QFormer 瓶颈压缩。学会利用相对相机几何是这个架构中最吃数据的部分,也是迁移性最好的部分 —— 比扩散头本身更值得迁移。预训练数据配比以 DROID 为主(带标定外参的真实 Franka 数据),所以一台类似的 7-DoF 机械臂 + 第三人称相机 + 腕部相机属于近域。

预训练收益最小的情况:如果你的控制频率、夹爪约定或相机布置与预训练数据差别很大,大部分先验就用不上了,从头训 + 增加迭代步数可能同样有竞争力。

注意视觉 backbone 在两种情况下都是冻结的预训练权重,所以"从头训"从来不意味着"从随机视觉特征开始"。

## 0. 前置条件

**标定好的相机外参是必需的,而且缺了会静默失败。** Context encoder 使用 PRoPE,它消费 `^{world}_{cam}T`,而且动作空间本身就定义在相机系下(`space_ee2cam`)。注意 PRoPE 和 RoPE 都是无参数的,所以切换 `pe_type` **不会**改变 `state_dict` —— checkpoint 依然会"加载成功",但位置编码的含义已经完全变了。症状:训练就是不收敛。录数据前先标定相机。

每一帧你需要有:

| 量 | 形状 | 说明 |
| --- | --- | --- |
| 末端位姿 | `(4, 4)` | `^{world}_{ee}T`,齐次矩阵 |
| 夹爪开合度 | 标量 | 归一化到 `[0, 1]`,0 = 闭合,1 = 张开 |
| 每路相机 RGB | `(H, W, 3)` | uint8 |
| 相机外参 | `(4, 4)` | `^{world}_{cam}T`,逐帧(腕部相机会动) |
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

## 3. 微调

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config finetune_real \
  --pretrained_ckpt ./checkpoints/E2VLA/PRETRAIN_EXP_NAME/ckpt_xxxxxxx.pt \
  -s MY_ROBOT_EXP
```

`finetune_real` 预设与 LIBERO 那几个的差别都源于数据量更小:20k 步而不是 70k,`max_lr=5e-5` 而不是 `1e-4`,开启 `grad_clip=1.0`,`bs=16`,EMA 从第 1k 步开启。`sample_multiplex=1000` 把 episode 列表膨胀,让一个 "epoch" 有可用的长度。此外它默认 `lora_rank=16`,见下一节。

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
- **`lora_rank > 0` 时必须传 `--pretrained_ckpt`**,否则直接拒绝运行:LoRA 冻结的基座权重如果本身是随机初始化的,等于 60% 的参数被永久卡在初始化状态。
- 部署时的 merge 必须在 EMA `copy_to` **之后**:EMA 的 shadow 参数就是 LoRA 因子本身,先 merge 会把 EMA 权重丢掉。

### 部署时的开销:零

`infer_utils/planner.py` 的 `load_model` 会自动读取 checkpoint 里的 `lora_rank`,注入、加载、然后调用 `merge_lora_linear` 把 `A @ B` 折进基座权重。merge 之后模型退回成一个普通的 `ActionExpert`,没有任何 `LoraLinear` 残留,推理没有额外算子和显存开销。整条链路(训练态输出 vs 部署态 merge 后输出)的数值漂移在 float32 下约 `1e-6`。

> 关于数值:LoRA 注入时 `B` 是零初始化的,所以 `lin(x) + (x @ A) @ B` 在函数意义上与 `lin(x)` 完全等价。但端到端测下来 context 输出会有约 `1e-6` 的漂移 —— 这不是逻辑错误:`lin(x) + 0` 会分配一个新张量,落在不同地址,可能走不同的 matmul/SDPA kernel 分块路径。已验证:把线程数钉到 1 时漂移显著减小,参数逐位相同,每个 `LoraLinear` 的输出与其基座 `Linear` 逐位相同。作为尺度参考,训练跑在 bfloat16 下,其 eps(~8e-3)比这个漂移大约 4000 倍。

## 4. 部署

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
controller.set_config("RealRobot")     # 按数据集类名索引
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
