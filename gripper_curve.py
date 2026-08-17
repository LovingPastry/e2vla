"""逐帧扫一条 episode，只看夹爪：GT / 预测 / 差值的曲线。

    # 第 0 条 episode 全程逐帧推理，出图 + 表
    CUDA_VISIBLE_DEVICES=0 python gripper_curve.py --ckpt ./checkpoints/E2VLA/VA_SA/ckpt_latest.pt

    # 换 episode / 只看一段 / 隔帧抽稀（快一点）
    CUDA_VISIBLE_DEVICES=0 python gripper_curve.py --ckpt ... --episode 3
    CUDA_VISIBLE_DEVICES=0 python gripper_curve.py --ckpt ... --start 80 --end 200 --stride 2

与 `dump_chunk.py` 的分工：那边是一个观测、整条 chunk 的所有通道，看"这一次预测了什么"；
这边是一条 episode、每帧只取 chunk 的**第一个动作**、只看夹爪那一维，看"夹爪指令在整段
轨迹上的走向"。后者才能回答"夹爪到底动没动"——单看一个 chunk，真值本来就可能整段都不翻转
（比如 episode 开头），什么也说明不了。

为什么只取 chunk 的第一个动作：闭环执行时每步重规划，真正被执行的就是这一个（`--exec_step`
可以改成看第 k 个，用来判断"是不是整个 chunk 都滞后"）。所以这条曲线与闭环时下发的夹爪
指令序列是同一个量，只是没有误差累积。

开环：观测全部来自数据集真值，预测不回灌。在训练集上跑时它先答的是拟合，不是泛化。

**画的是哪一层的数**：数据集 state 空间的开合度，契约 `[0 (闭合), 1 (张开)]`——既没做
`(x-0.5)*2` 的重标定，也没过 q01/q99 归一化。GT 直接取 `future_actions[..., -1]`，pred 是
模型输出经 `action2states` 解码回来的（那里做的正是 `/2 + 0.5`，开了 action_norm 则先
unnormalize）。所以这条曲线与 `planner` 输出、与 `eval.py` 里 `>0.5` 二值化看到的是同一个
量。想看模型内部那一层（t3r6 / 关节 + `[-1,1]` 的夹爪 / 归一化后的值）用
`dump_chunk.py --raw`；两者差一个 `x = 值/2 + 0.5`，例如 action 空间的 -1.0 ~ -0.4 对应
这里的 0.0 ~ 0.3。

模型加载走 `infer_utils.planner.load_model`（三个戳的校验、LoRA 注入顺序、EMA copy_to），
数据装配走 `dump_chunk.build_datasets`（相机顺序、padding 与 test.py / dump_chunk.py 一致）。
"""

import os
import csv
import random
import argparse
import unicodedata
from typing import Dict, List, Optional, Tuple

import envars  # noqa: F401  必须在 torchvision（被 data_utils 带进来）之前 import
import h5py
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")  # 无显示环境；必须在 pyplot 之前
import matplotlib.pyplot as plt

from test import flatten_valid
from dump_chunk import build_datasets
from infer_utils.planner import load_model, parse_config


RESULT_DIR = "./eval_results"

# 下游把连续开合度变成指令时用的阈值：`examples/libero/eval.py` 是 `(g > 0.5)` 之后
# 映射成 -1/+1。曲线跨不跨这条线，比它抖多少更能决定夹爪动不动。
BINARY_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# episode 定位
# ---------------------------------------------------------------------------

def resolve_episode(ds_list: List, index: int) -> Tuple[object, int, int]:
    """全局 episode 序号 -> (所属数据集, 数据集内序号, 全局总数)。

    配了多个数据集类时按 `ConcatDataset` 的顺序拼接，与 `test.py` / `dump_chunk.py`
    的编号空间一致。
    """
    total = sum(len(d) for d in ds_list)
    assert 0 <= index < total, "--episode 越界：可选 [0, {})".format(total)
    offset = index
    for d in ds_list:
        if offset < len(d):
            return d, offset, total
        offset -= len(d)
    raise AssertionError("unreachable")


def episode_length(dataset, local_index: int) -> int:
    """这条 episode 有多少帧。

    两条数据路径的存储方式不同，但 `last_obs_index` 的取值范围都是 [0, L)：
    memmap 那边是 `ee_poses` 的长度（`RealBinDataset.sample_traj`），h5 那边是
    `ee_pose` 数据集的长度（`DataSampler.sample_hdf5` 第一行就是它）。
    """
    if hasattr(dataset, "load_traj"):  # RealBinDataset
        return len(dataset.load_traj(local_index)["ee_poses"])
    with h5py.File(dataset.h5_filelist[local_index], "r") as h5:
        return int(h5["ee_pose"].len())


def sample_at(dataset, local_index: int, frame: int, seed: int) -> Dict:
    """把 chunk 的锚点钉在第 `frame` 帧，取一个样本。

    `debug_sample_index` 是实例属性而不是 `__getitem__` 的参数，见
    `H5DatasetMapBase.__init__`：各子类都重写了 `__getitem__` 做后处理（Libero 改
    prompt、Droid 沿夹爪轴平移 ee_pose），绕开它们喂给模型的就不是训练时那份数据。

    种子同时盖住 `random`（h5 那侧多候选 prompt 是随机抽的）和 `np.random`，这样同一帧
    重跑两次拿到的是同一个样本；帧号进种子，是为了不同帧之间不共用一次抽样。
    """
    random.seed(seed * 1000003 + frame)
    np.random.seed((seed * 1000003 + frame) % (2 ** 31 - 1))
    dataset.debug_sample_index = int(frame)
    try:
        return dataset[local_index]
    finally:
        # 复原，免得同一个实例被后续代码当成正常训练集用
        dataset.debug_sample_index = None


def collate(samples: List[Dict], device: str) -> Dict:
    """一批样本 -> 模型输入。手写而不用 `default_collate`：只需要 stack，而 collate
    对 `prompt_text` 这类非张量字段的处理随 torch 版本变。"""
    batch = {}
    for k, v in samples[0].items():
        if isinstance(v, np.ndarray):
            batch[k] = torch.from_numpy(np.stack([s[k] for s in samples])).to(device)
        elif isinstance(v, Tensor):
            batch[k] = torch.stack([s[k] for s in samples]).to(device)
        else:
            batch[k] = [s[k] for s in samples]  # prompt_text: List[str]
    return batch


# ---------------------------------------------------------------------------
# 取夹爪那一维
# ---------------------------------------------------------------------------

def first_valid_rows(mask: Tensor) -> List[int]:
    """每个样本在 `flatten_valid` 结果里的首个有效 ee 所在行。

    展平是按 (B, Nee) 行主序走的（和 `ActionExpert.forward` 里的 `batch_index` 同序），
    所以每个样本的起始行就是它前面所有样本的有效 ee 数之和。单臂时就是 0..B-1。
    """
    counts = mask.sum(dim=-1).tolist()
    rows, acc = [], 0
    for c in counts:
        rows.append(acc)
        acc += int(c)
    return rows


def gripper_of(states: Tensor, rows: List[int], step: int) -> np.ndarray:
    """(B', Ta, state_dim) -> (B,)：每个样本、chunk 第 `step` 步、夹爪那一维。

    两个动作空间都把夹爪放在 state 的最后一维（EE 是 16+1，关节是 nq+1）。
    """
    return states[rows, step, -1].float().cpu().numpy()


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def print_table(frames: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                delta: np.ndarray):
    print()
    print("  {:>7}{:>10}{:>10}{:>10}   {}".format(
        "frame", "GT", "pred", "delta", "GT|pred 二值"))
    print("  " + "-" * 56)
    for f, g, p, d in zip(frames, gt, pred, delta):
        print("  {:>7d}{:>10.3f}{:>10.3f}{:>+10.3f}   {}|{}".format(
            int(f), g, p, d,
            int(g > BINARY_THRESHOLD), int(p > BINARY_THRESHOLD)))


def flips(x: np.ndarray) -> int:
    b = x > BINARY_THRESHOLD
    return int(np.count_nonzero(b[1:] != b[:-1]))


def pad(label: str, width: int) -> str:
    """左对齐到 `width` 个**显示**列。

    `"{:<14}".format(...)` 数的是码位，中文是双宽字符，混排时列会歪掉。
    """
    shown = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in label)
    return label + " " * max(0, width - shown)


def print_summary(gt: np.ndarray, pred: np.ndarray, delta: np.ndarray):
    agree = float(np.mean((gt > BINARY_THRESHOLD) == (pred > BINARY_THRESHOLD)))
    gt_range, pred_range = float(np.ptp(gt)), float(np.ptp(pred))
    print()
    print("  {}{:>9}{:>9}".format(pad("", 14), "GT", "pred"))
    print("  {}{:>9.3f}{:>9.3f}".format(pad("min", 14), gt.min(), pred.min()))
    print("  {}{:>9.3f}{:>9.3f}".format(pad("max", 14), gt.max(), pred.max()))
    print("  {}{:>9.3f}{:>9.3f}".format(pad("极差", 14), gt_range, pred_range))
    print("  {}{:>9d}{:>9d}".format(pad("过阈值翻转", 14), flips(gt), flips(pred)))
    print()
    print("  MAE {:.4f}   最大偏差 {:.4f}   二值一致率 {:.1%}".format(
        float(np.abs(delta).mean()), float(np.abs(delta).max()), agree))
    # 数据集契约是 [0 (闭合), 1 (张开)]（dataset_base.py 的 sample_hdf5 文档）。越界不会
    # 在训练里报错，只会让 `states2action` 的 (x-0.5)*2 把动作推到 [-1,1] 之外，然后
    # 归一化统计、loss 权重、二值化阈值全部按错误的量程工作。
    if gt.min() < -1e-6 or gt.max() > 1 + 1e-6:
        print("  [WARN] GT 夹爪超出 [0,1]（实测 {:.3f} ~ {:.3f}）。这一维不是开合度，或者"
              "归一化常数不对：".format(gt.min(), gt.max()))
        print("         跑 `python -m data_utils.dataset_real` 让 check_contract 定位；"
              "memmap 数据看 RealBinDataset.get_openness 的 GRIPPER_MIN/MAX 与 "
              "norm_openness 分支（后者原样透传，不做 clip）。")
    elif gt.max() < 0.6:
        print("  [INFO] GT 夹爪最大为 {:.3f}，示教未覆盖物理全开。这本身不是"
              "数据错误；物理量程端点正确时不要重定标。".format(gt.max()))
        print("         如果部署端使用 `>{:.1f}` 二值化，它才会被错误地变成"
              "恒定全闭指令；真机应直接执行连续位置。".format(BINARY_THRESHOLD))
    if pred_range < 0.1 and gt_range > 0.3:
        print("  [WARN] 真值在动（极差 {:.3f}）而预测几乎是常量（极差 {:.3f}）——"
              "夹爪塌成了边缘均值".format(gt_range, pred_range))
    elif flips(pred) == 0 and flips(gt) > 0:
        print("  [WARN] 预测有起伏但从不跨过 {:.2f}，二值化后仍是恒定指令——"
              "下游看到的依然是'夹爪不动'".format(BINARY_THRESHOLD))


def plot(frames: np.ndarray, gt: np.ndarray, pred: np.ndarray, delta: np.ndarray,
         out_path: str, title: str):
    """GT / pred 共用左轴（同一尺度才能直接比大小），delta 单独走右轴。

    图里一律用英文：matplotlib 的默认字体没有 CJK 字形，中文标签会渲染成方块。
    """
    fig, ax_val = plt.subplots(figsize=(12, 5))

    ax_val.plot(frames, gt, color="#1f77b4", lw=1.8, label="GT")
    ax_val.plot(frames, pred, color="#ff7f0e", lw=1.5, label="pred")
    ax_val.set_xlabel("frame index")
    ax_val.set_ylabel("gripper openness  [0 = close, 1 = open]")
    ax_val.grid(alpha=0.25)

    # 默认锁成开合度的定义域：自动缩放会把一条几乎水平的预测线放大成剧烈震荡。但**只在
    # 数据确实落在里面时**才锁——否则就是拿一个漂亮的坐标轴把越界的点裁出画面，看图的人
    # 完全不会发现。超出就扩轴 + 在图上写明。
    lo, hi = -0.05, 1.05
    data_lo = float(min(gt.min(), pred.min()))
    data_hi = float(max(gt.max(), pred.max()))
    clipped = data_lo < lo or data_hi > hi
    if clipped:
        lo, hi = min(lo, data_lo - 0.05), max(hi, data_hi + 0.05)
        ax_val.text(0.01, 0.02,
                    "note: data outside [0, 1] -> axis widened to "
                    "[{:.2f}, {:.2f}]".format(lo, hi),
                    transform=ax_val.transAxes, fontsize=8, color="#d62728")
    ax_val.set_ylim(lo, hi)

    ax_err = ax_val.twinx()
    span = float(np.abs(delta).max())
    span = span * 1.15 if span > 1e-6 else 1e-3  # 全零时给个非退化的范围
    ax_err.plot(frames, delta, color="#d62728", lw=1.2, alpha=0.7,
                label="delta (pred - GT)")
    ax_err.set_ylabel("delta (pred - GT)", color="#d62728")

    # 左轴锁在 [-0.05, 1.05] 且右轴关于 0 对称时，两者的中线都落在图高正中间——这时画
    # 两条参考线只会重叠成一条颜色可疑的线，合成一条并把两个含义都写进图例。扩轴之后这个
    # 巧合就没了，必须分开画。
    if abs((lo + hi) / 2 - BINARY_THRESHOLD) < 1e-9:
        ax_val.axhline(BINARY_THRESHOLD, color="gray", ls=":", lw=1.2,
                       label="openness {:.1f}  =  delta 0".format(BINARY_THRESHOLD))
    else:
        ax_val.axhline(BINARY_THRESHOLD, color="gray", ls=":", lw=1.2,
                       label="openness {:.1f} (binarize)".format(BINARY_THRESHOLD))
        ax_err.axhline(0.0, color="#d62728", ls="--", lw=0.8, alpha=0.45)
    ax_err.tick_params(axis="y", labelcolor="#d62728")
    ax_err.set_ylim(-span, span)  # 对称，0 才落在正中间

    handles = ax_val.get_legend_handles_labels()
    errs = ax_err.get_legend_handles_labels()
    ax_val.legend(handles[0] + errs[0], handles[1] + errs[1],
                  loc="upper right", fontsize=9, framealpha=0.9)
    ax_val.set_title(title, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_csv(path: str, frames: np.ndarray, gt: np.ndarray, pred: np.ndarray,
             delta: np.ndarray):
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["frame", "gt", "pred", "delta"])
        for row in zip(frames.tolist(), gt.tolist(), pred.tolist(), delta.tolist()):
            writer.writerow(row)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="逐帧扫一条 episode，只看夹爪的 GT / 预测 / 差值",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ckpt", type=str, required=True,
                        help="ckpt 路径；同目录下的 *.json 提供模型与数据配置")
    parser.add_argument("--dataset", type=str, nargs="+", default=None,
                        help="数据集类名，默认用 ckpt config 里记的（python datavis.py -l 列全部）")
    parser.add_argument("--data_root", type=str, default=None,
                        help="仅对 inst() 接受 data_root 的数据集有效（如 RealBinDataset）")
    parser.add_argument("--episode", type=int, default=0, help="第几条 episode")
    parser.add_argument("--start", type=int, default=0, help="起始帧")
    parser.add_argument("--end", type=int, default=None, help="结束帧（不含），默认到末尾")
    parser.add_argument("--stride", type=int, default=1, help="每隔几帧推理一次")
    parser.add_argument("--exec_step", type=int, default=0,
                        help="取 chunk 里的第几个动作；0 = 闭环时真正被执行的那个")
    parser.add_argument("--bs", type=int, default=8, help="一次前向塞几帧")
    parser.add_argument("--seed", type=int, default=0,
                        help="只影响 h5 多候选 prompt 的抽样；锚点由 --start/--stride 定死")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ema", action="store_true",
                        help="用 EMA 影子权重，仅当训练时真开了 ema_enabled")
    parser.add_argument("--fp32", action="store_true",
                        help="关掉 bfloat16，默认跟随 ckpt config 的 fp16")
    parser.add_argument("--out", type=str, default=None,
                        help="图片路径，默认 ./eval_results/gripper_<ckpt>_ep<N>.png")
    parser.add_argument("--no_table", action="store_true", help="不打印逐帧表格")
    parser.add_argument("--pad_ncam", type=int, default=0, help="0 表示按训练配置推断")
    parser.add_argument("--pad_nee", type=int, default=0, help="0 表示按训练配置推断")
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    assert args.stride >= 1 and args.bs >= 1 and args.start >= 0 and args.exec_step >= 0

    cfg, _ = parse_config(os.path.dirname(os.path.abspath(args.ckpt)))
    model, _ = load_model(args.ckpt, args.device, use_ema=args.ema)
    action_space = model.action_space
    fp16 = cfg.fp16 and not args.fp32

    print("=" * 72)
    print("[INFO] ckpt {}".format(os.path.basename(args.ckpt)))
    print("[INFO] objective={}  action_space={}  context_encoder={}  采样步数={}".format(
        cfg.objective, action_space.name, cfg.context_encoder,
        model.actor.inference_timesteps))

    ds_list = build_datasets(cfg, args)
    dataset, local_index, total = resolve_episode(ds_list, args.episode)
    length = episode_length(dataset, local_index)
    end = length if args.end is None else min(args.end, length)
    assert args.start < end, "帧区间为空：start={} end={}（episode 共 {} 帧）".format(
        args.start, end, length)
    assert args.exec_step < dataset.config.num_future_states, \
        "--exec_step 必须小于 chunk 长度 {}".format(dataset.config.num_future_states)

    frames = np.arange(args.start, end, args.stride)
    print("[INFO] episode {}/{} -> {}[{}]，共 {} 帧，推理 {} 帧（stride={}）".format(
        args.episode, total, type(dataset).__name__, local_index, length,
        len(frames), args.stride))
    print("[INFO] 取 chunk 的第 {} 个动作（chunk 长 {}，采样间隔 {} 帧）".format(
        args.exec_step, dataset.config.num_future_states,
        dataset.config.sample_state_gaps))

    gt_list, pred_list, kept_frames = [], [], []
    for begin in tqdm(range(0, len(frames), args.bs), desc="frames", unit="batch",
                      dynamic_ncols=True):
        chunk_frames = frames[begin:begin + args.bs]
        samples = [sample_at(dataset, local_index, int(f), args.seed)
                   for f in chunk_frames]
        batch = collate(samples, args.device)

        pred = model(
            rgbs=batch["rgbs"],
            obs_norm_xys=batch["obs_norm_xys"],
            obs_extrinsics=batch["obs_extrinsics"],
            prompt_text=batch["prompt_text"],

            ee_poses=batch["ee_poses"],
            history_actions=batch["history_actions"],
            future_actions=batch["future_actions"],  # inference 下只用来推形状
            valid_ee_mask=batch["valid_ee_mask"],
            inference=True,
            fp16=fp16,
        )  # (B, Ta, Nee, state_dim)

        mask = batch["valid_ee_mask"]
        rows = first_valid_rows(mask)
        gt_states = flatten_valid(batch["future_actions"], mask)
        pred_states = flatten_valid(pred, mask)
        if gt_states.shape[0] == 0:
            tqdm.write("[WARN] 帧 {} 起的这批没有有效 ee，跳过".format(chunk_frames[0]))
            continue
        if not torch.isfinite(pred_states).all():
            # 静默跳过会让曲线凭空变好看
            tqdm.write("[WARN] 帧 {} 起的这批预测含 NaN/Inf，跳过".format(chunk_frames[0]))
            continue

        gt_list.append(gripper_of(gt_states, rows, args.exec_step))
        pred_list.append(gripper_of(pred_states, rows, args.exec_step))
        kept_frames.append(np.asarray(chunk_frames))

    if not kept_frames:
        print("[ERROR] 没有任何一帧成功推理")
        return

    frames = np.concatenate(kept_frames)
    gt = np.concatenate(gt_list)
    pred = np.concatenate(pred_list)
    delta = pred - gt

    if not args.no_table:
        print_table(frames, gt, pred, delta)
    print_summary(gt, pred, delta)

    os.makedirs(RESULT_DIR, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.ckpt))[0]
    out_path = args.out or os.path.join(
        RESULT_DIR, "gripper_{}_ep{}.png".format(tag, args.episode))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    title = "{} | {}[{}] | episode {} | chunk step {} | objective {}".format(
        tag, type(dataset).__name__, local_index, args.episode, args.exec_step,
        cfg.objective)
    plot(frames, gt, pred, delta, out_path, title)
    csv_path = os.path.splitext(out_path)[0] + ".csv"
    save_csv(csv_path, frames, gt, pred, delta)

    print()
    print("[INFO] 图 -> {}".format(out_path))
    print("[INFO] 数 -> {}".format(csv_path))
    print("=" * 72)
    print("怎么读：左轴是 GT 与 pred（同一尺度，直接比高低），右轴是 delta（对称、0 居中）。")
    print("  * pred 是一条几乎水平的线 -> 夹爪塌成了常量，与观测无关；")
    print("  * pred 跟着 GT 起伏但整体不跨灰色虚线 -> 二值化后仍是恒定指令，照样不动；")
    print("  * pred 跟得上但整体右移几帧 -> 是滞后不是不动，`--exec_step` 调大能看出来。")
    print("=" * 72)


if __name__ == "__main__":
    main()
