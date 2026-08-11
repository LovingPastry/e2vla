"""开环测试：用训练好的 checkpoint 在已有数据上跑完整采样循环，比对预测动作块与真值。

    # 最简：数据集、动作空间、LoRA、归一化全部从 ckpt 目录的 config json 里读
    CUDA_VISIBLE_DEVICES=0 python test.py --ckpt ./checkpoints/E2VLA/FT_EXP/ckpt_0070000.pt

    # 换数据集 / 加大样本量 / 用 EMA 权重（只有训练时真开了 EMA 才能加 --ema）
    CUDA_VISIBLE_DEVICES=0 python test.py --ckpt ... --dataset Libero10 --num_samples 512 --ema

    # 存逐样本误差 + 画前 8 个样本的预测轨迹叠加图
    CUDA_VISIBLE_DEVICES=0 python test.py --ckpt ... --save --vis 8

这是 loss 曲线和仿真 rollout 之间缺的那一档：

- 训练 loss 衡量的是"单步去噪回归得准不准"，量纲被 objective 和通道权重搅在一起，
  ddim 与 flow 的数值根本不可比（见 `models/action_space.py:ActionSpace.loss`）；
- 仿真 rollout 要另起三个进程、换一个 conda 环境，而且只有 LIBERO 有环境；
- 这里跑的是**完整采样循环**，误差单位是米 / 度 / 弧度，与 rollout 成功率同量纲、
  跨 objective 可比，也是唯一一个不开仿真就能读出"这个 ckpt 大概能不能用"的数。

与 `train.py` 的 `log_sample_interval`（`sample/*` 曲线）算的是同一个量，区别只在于那边
量的是当前训练 batch、只出一个标量，这边是遍历数据集、可复现、还给出误差沿 chunk 的分布。

"开环"的含义：每个样本都从**真值观测**出发预测一个 chunk，预测不会反过来影响下一帧观测。
所以它回答的是"采样器能不能复现数据里的动作"，不回答"误差会不会在闭环里累积"——后者只有
仿真或真机能答。在训练集上跑同理，它先答的是拟合，不是泛化。

模型构建与权重加载整个交给 `infer_utils.planner.load_model`：LoRA 的注入顺序、EMA 的
copy_to 与 merge 的先后、objective / action_layout / action_norm 三个戳的校验都在那里，
在这里重写一份迟早会和部署路径走岔——而走岔的表现是权重干净加载然后输出垃圾。
"""

import os
import json
import time
import random
import inspect
import argparse
from typing import Dict, List, Optional

import envars  # noqa: F401  必须在 torchvision（被 data_utils 带进来）之前 import
import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

from configs import TrainConfig
from data_utils import datasets as datasets_mod
from data_utils.dataset_base import H5DatasetMapBase
from infer_utils.planner import load_model, parse_config
from infer_utils.draw_traj import visualize_traj


RESULT_DIR = "./eval_results"


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------

class FixedSampleView(Dataset):
    """把底层数据集"每次取样随机挑一个时刻"的行为，固定成由样本序号唯一决定的时刻。

    数据集的 `__getitem__(i)` 只指定 episode，episode 内取哪一帧是 `np.random.choice`
    抽的（`DataSampler.sample_hdf5` / `RealBinDataset.sample_indices`）。不固定住它，
    两次评测就是在两批不同的样本上比大小，几个百分点的差异读不出意义。

    这里只在调用前把 RNG 定住，而不是走 `debug_sample_index`：后者要绕开子类自己的
    `__getitem__`（Libero 改 prompt、Droid 把 ee_poses 沿夹爪轴平移 15cm、PickPlaceCan
    换 prompt 模板），绕开之后喂给模型的就不是训练时的那份数据了；而且 `sample_hdf5`
    每次都会为它打印一行。种子同时盖住 `random`（prompt 抽样、相机顺序）和 `np.random`
    （时刻），所以整个样本都是 (seed, k) 的确定函数，与 num_workers 无关。

    `repeats > 1` 时同一条 episode 会在不同时刻被采多次；k 按 repeat 分块编号，
    截断到前 N 个样本仍然覆盖到所有 episode，而不是把前几条 episode 反复采。
    """

    def __init__(self, dataset: Dataset, repeats: int = 1, seed: int = 0):
        assert repeats >= 1
        self.dataset = dataset
        self.repeats = repeats
        self.seed = seed

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, k: int):
        episode_index = k % len(self.dataset)
        item_seed = (self.seed * 1000003 + k) % (2 ** 31 - 1)
        random.seed(item_seed)
        np.random.seed(item_seed)
        torch.manual_seed(item_seed)

        out = self.dataset[episode_index]
        # 只有本脚本读，模型的 forward 是按关键字取参的，多一个键不影响它
        out["eval_index"] = k
        out["episode_index"] = episode_index
        return out


def resolve_dataset_classes(cfg: TrainConfig, names: Optional[List[str]]):
    """--dataset 给的类名，或者 ckpt 自己 config json 里记的那几个。"""
    if not names:
        assert len(cfg.dataset_classes) > 0, (
            "ckpt 的 config json 里没有 dataset_classes，请用 --dataset 显式指定。"
            "可选项：python datavis.py -l")
        return list(cfg.dataset_classes)

    classes = []
    for name in names:
        D = getattr(datasets_mod, name, None)
        if not (isinstance(D, type) and issubclass(D, H5DatasetMapBase)):
            raise ValueError(
                "未知的数据集类 '{}'。可选项见 `python datavis.py -l`".format(name))
        classes.append(D)
    return classes


def instantiate(D: type, data_root: Optional[str]):
    """`inst()` 的签名各类不同：RealBinDataset 收 data_root，h5 那几个类是写死的 glob。"""
    if data_root and "data_root" in inspect.signature(D.inst).parameters:
        return D.inst(data_root=data_root)
    if data_root:
        print("[WARN] {}.inst() 不接受 data_root，--data_root 对它无效（它按 "
              "./data_converted/ 下的相对路径 glob，得从 e2vla/ 目录跑）".format(D.__name__))
    return D.inst()


def build_dataloader(cfg: TrainConfig, args):
    classes = resolve_dataset_classes(cfg, args.dataset)
    print("[INFO] 测试数据集: {}".format(", ".join(D.__name__ for D in classes)))

    # 相机顺序在部署时是不打乱的（`planner.parse_config` 也这么干）：camera_names[0]
    # 定义了整个动作坐标系，评测要复现的是部署条件，不是训练时的数据增广。
    for D in classes:
        if D.config.shuffle_cameras:
            print("[INFO] {}: shuffle_cameras True -> False（对齐部署条件）"
                  .format(D.__name__))
            D.config.shuffle_cameras = False

    ds_list = [instantiate(D, args.data_root) for D in classes]

    # padding 要对齐**训练时**的最大相机 / ee 数，而不是本次测试子集的。训练时
    # `concat_datasets` 取的是 cfg 里所有数据集的 max，模型见过的就是那个宽度（多出来
    # 的相机是零填充的）；只测其中一个数据集时按子集重算，输入宽度就变了，而这既不会
    # 报错也不会缺 key，只会让数字偏低。相机数是类属性，不需要把数据都实例化出来。
    train_classes = cfg.dataset_classes or classes
    pad_ncam = args.pad_ncam or max(len(D.config.camera_names) for D in train_classes)
    pad_nee = args.pad_nee or max(len(D.config.ee_indices) for D in train_classes)
    for d in ds_list:
        d.pad2ncam = pad_ncam
        d.pad2nee = pad_nee
    print("[INFO] pad2ncam={}, pad2nee={}（取自训练配置 {}）".format(
        pad_ncam, pad_nee, [D.__name__ for D in train_classes]))

    dataset = ds_list[0] if len(ds_list) == 1 else ConcatDataset(ds_list)
    view = FixedSampleView(dataset, repeats=args.repeats, seed=args.seed)

    num_samples = len(view) if args.num_samples < 0 else min(args.num_samples, len(view))
    if num_samples < len(view):
        view = torch.utils.data.Subset(view, list(range(num_samples)))
    print("[INFO] {} 条 episode x {} 次采样 = {} 个样本，实测 {} 个".format(
        len(dataset), args.repeats, len(dataset) * args.repeats, num_samples))

    loader = DataLoader(
        dataset=view,
        batch_size=args.bs,
        num_workers=args.workers,
        shuffle=False,          # 顺序固定，逐样本误差才对得上 episode
        persistent_workers=False,
    )
    return loader, ds_list, num_samples


# ---------------------------------------------------------------------------
# 误差
# ---------------------------------------------------------------------------

def flatten_valid(x: Tensor, mask: Tensor) -> Tensor:
    """(B, T, Nee, D) -> (B', T, D)，与模型内部对 valid ee 的展平方式一致。

    无效槽位里装的是动作空间的占位填充（EE 空间是单位阵），算进去纯属噪声。
    """
    return x.transpose(1, 2)[mask]


def hold_baseline(history_actions: Tensor, mask: Tensor, horizon: int) -> Tensor:
    """"原地不动"的参照系：把最后一帧历史状态沿整个 chunk 重复。

    没有它，`pos_err_m = 0.018` 这种数是读不出好坏的——它取决于这个数据集在
    `num_future_states * sample_state_gaps` 步里本来能走多远。这条基线的误差恰好就是
    chunk 的真实位移量级，策略的误差得明显小于它才说明学到了东西。
    """
    last = flatten_valid(history_actions, mask)[:, -1:]   # (B', 1, D)
    return last.expand(-1, horizon, -1).contiguous()


def per_sample_errors(action_space, pred_states: Tensor, gt_states: Tensor):
    """逐样本调用 `ActionSpace.state_error`，返回 List[Dict[str, float]]。

    逐样本而不是整批一次：整批只给一个均值，看不到分位数，也定位不到是哪条 episode 拖
    的后腿。误差的定义（旋转角的 clamp、夹爪的阈值）一律走动作空间自己的实现，这里不
    另写一份——那正是最容易和训练侧对不上的地方。
    """
    return [action_space.state_error(pred_states[j:j + 1], gt_states[j:j + 1])
            for j in range(pred_states.shape[0])]


def horizon_curve(action_space, pred_states: Tensor, gt_states: Tensor):
    """误差沿 chunk 位置的分布：对每个 t 单独调一次 state_error。

    真正决定闭环表现的是 chunk 末端——执行到那里时下一次重规划还没发生。均值会把它摊平。
    """
    horizon = pred_states.shape[1]
    return [action_space.state_error(pred_states[:, t:t + 1], gt_states[:, t:t + 1])
            for t in range(horizon)]


def summarize(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = rows[0].keys() if rows else []
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = {
            "mean": float(v.mean()),
            "p50": float(np.percentile(v, 50)),
            "p90": float(np.percentile(v, 90)),
            "max": float(v.max()),
        }
    return out


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def save_overlay(cpu_batch: Dict, pred: Tensor, j: int, path: str):
    """把真值 chunk（绿）和预测 chunk（红）投影回最新一帧图像。

    只对 EE 动作空间有意义（关节角要正运动学才能投影），而且数据集得有真实外参——
    全 identity 时 `visualize_traj` 会走 `is_identity_extrinsics` 分支只返回原图。
    """
    sample = {k: v[j] for k, v in cpu_batch.items() if isinstance(v, (Tensor, list))}
    bgr = visualize_traj(
        data=sample,
        future_ee_states=[sample["future_actions"], pred[j].float().cpu()],
        colors=[(0, 255, 0), (0, 0, 255)],
    )
    if bgr.dtype == np.float32:
        bgr = (bgr * 255.).clip(0, 255).astype(np.uint8)
    cv2.imwrite(path, bgr)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="开环测试：在已有数据上比对采样出的动作块与真值",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ckpt", type=str, required=True,
                        help="ckpt 路径；同目录下的 *.json 提供模型与数据配置")
    parser.add_argument("--dataset", type=str, nargs="+", default=None,
                        help="数据集类名，默认用 ckpt config 里记的那几个（python datavis.py -l 列全部）")
    parser.add_argument("--data_root", type=str, default=None,
                        help="仅对 inst() 接受 data_root 的数据集有效（如 RealBinDataset）")
    parser.add_argument("--num_samples", type=int, default=256,
                        help="评测样本数，-1 表示 len(dataset) * repeats 全跑")
    parser.add_argument("--repeats", type=int, default=1,
                        help="每条 episode 在不同时刻采样几次")
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0,
                        help="决定采到哪些时刻；跨 ckpt 对比时必须保持一致")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ema", action="store_true",
                        help="用 EMA 影子权重，仅当训练时真的开了 ema_enabled")
    parser.add_argument("--fp32", action="store_true",
                        help="关掉 bfloat16，默认跟随 ckpt config 的 fp16")
    parser.add_argument("--pad_ncam", type=int, default=0, help="0 表示按训练配置推断")
    parser.add_argument("--pad_nee", type=int, default=0, help="0 表示按训练配置推断")
    parser.add_argument("--vis", type=int, default=0,
                        help="保存前 N 个样本的轨迹叠加图（绿=真值，红=预测）")
    parser.add_argument("--save", action="store_true",
                        help="把汇总与逐样本误差写到 ./eval_results/")
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()

    cfg, _ = parse_config(os.path.dirname(os.path.abspath(args.ckpt)))
    # 权重加载的全部规矩都在 load_model 里，包括三个戳的校验。它内部会再 parse 一次
    # config json，所以上面那几行 [INFO] 会重复一遍。
    model, _ = load_model(args.ckpt, args.device, use_ema=args.ema)
    action_space = model.action_space
    fp16 = cfg.fp16 and not args.fp32
    print("[INFO] fp16={}, objective={}, action_space={}".format(
        fp16, cfg.objective, action_space.name))

    loader, ds_list, num_samples = build_dataloader(cfg, args)

    can_visualize = args.vis > 0 and action_space.layout == "cam_rel_t3r6_openness"
    if args.vis > 0 and not can_visualize:
        print("[WARN] --vis 只支持 EE 动作空间（关节角要正运动学才能投影回图像），已跳过")
    vis_dir = os.path.join(RESULT_DIR, "openloop_vis")
    if can_visualize:
        os.makedirs(vis_dir, exist_ok=True)

    policy_rows: List[Dict[str, float]] = []
    hold_rows: List[Dict[str, float]] = []
    curve_sum: List[Dict[str, float]] = []
    curve_count = 0
    index_rows: List[Dict] = []
    n_vis = 0
    forward_seconds = 0.0

    pbar = tqdm(loader, desc="open-loop", unit="batch", dynamic_ncols=True)
    for cpu_batch in pbar:
        batch = {k: (v.to(args.device, non_blocking=True) if isinstance(v, Tensor) else v)
                 for k, v in cpu_batch.items()}

        start = time.perf_counter()
        pred = model(
            rgbs=batch["rgbs"],
            obs_norm_xys=batch["obs_norm_xys"],
            obs_extrinsics=batch["obs_extrinsics"],
            prompt_text=batch["prompt_text"],

            ee_poses=batch["ee_poses"],
            history_actions=batch["history_actions"],
            future_actions=batch["future_actions"],   # inference 下只用来推形状
            valid_ee_mask=batch["valid_ee_mask"],
            inference=True,
            fp16=fp16,
        )  # (B, Ta, Nee, state_dim)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        forward_seconds += time.perf_counter() - start

        mask = batch["valid_ee_mask"]
        pred_states = flatten_valid(pred, mask).float()
        gt_states = flatten_valid(batch["future_actions"], mask).float()
        if pred_states.shape[0] == 0:
            tqdm.write("[WARN] 整个 batch 没有有效 ee，跳过")
            continue
        if not torch.isfinite(pred_states).all():
            # 静默跳过会让均值凭空变好，所以要喊出来
            tqdm.write("[WARN] 预测里有 NaN/Inf，该 batch 不计入统计")
            continue

        policy_rows.extend(per_sample_errors(action_space, pred_states, gt_states))
        hold_states = hold_baseline(batch["history_actions"].float(), mask,
                                    gt_states.shape[1])
        hold_rows.extend(per_sample_errors(action_space, hold_states, gt_states))

        # 逐 t 的曲线按 batch 累加（每个 batch 内 B' 不同，按样本数加权）
        batch_curve = horizon_curve(action_space, pred_states, gt_states)
        weight = pred_states.shape[0]
        if not curve_sum:
            curve_sum = [{k: 0.0 for k in batch_curve[0]} for _ in batch_curve]
        for t, step in enumerate(batch_curve):
            for k, v in step.items():
                curve_sum[t][k] += v * weight
        curve_count += weight

        for j in range(len(cpu_batch["eval_index"])):
            index_rows.append({
                "eval_index": int(cpu_batch["eval_index"][j]),
                "episode_index": int(cpu_batch["episode_index"][j]),
                "prompt_text": cpu_batch["prompt_text"][j],
            })

        if can_visualize and n_vis < args.vis:
            for j in range(min(pred.shape[0], args.vis - n_vis)):
                save_overlay(cpu_batch, pred, j,
                             os.path.join(vis_dir, "sample_{:04d}.png".format(n_vis)))
                n_vis += 1

        if policy_rows:
            first_key = list(policy_rows[0].keys())[0]
            pbar.set_postfix_str("{} = {:.4f}".format(
                first_key, float(np.mean([r[first_key] for r in policy_rows]))))

    if not policy_rows:
        print("[ERROR] 没有任何样本进入统计")
        return

    policy = summarize(policy_rows)
    hold = summarize(hold_rows)
    curve = [{k: v / curve_count for k, v in step.items()} for step in curve_sum]

    report(args, cfg, action_space, policy, hold, curve, len(policy_rows),
           forward_seconds, model.actor.inference_timesteps)

    if args.save:
        os.makedirs(RESULT_DIR, exist_ok=True)
        tag = os.path.splitext(os.path.basename(args.ckpt))[0]
        names = "-".join(type(d).__name__ for d in ds_list)
        out_path = os.path.join(RESULT_DIR, "openloop_{}_{}.json".format(tag, names))
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump({
                "ckpt": os.path.abspath(args.ckpt),
                "ema": args.ema,
                "datasets": [type(d).__name__ for d in ds_list],
                "objective": cfg.objective,
                "action_space": action_space.name,
                "action_layout": action_space.layout,
                "num_samples": len(policy_rows),
                "seed": args.seed,
                "repeats": args.repeats,
                "policy": policy,
                "hold_baseline": hold,
                "horizon_curve": curve,
                # 多臂（pad2nee > 1）时 policy_rows 是按 (样本, 有效 ee) 展平的，行数
                # 与 index_rows 对不上，这时只留误差，不硬凑 episode 归属
                "per_sample": ([dict(meta, **row)
                                for meta, row in zip(index_rows, policy_rows)]
                               if len(index_rows) == len(policy_rows) else policy_rows),
            }, fp, ensure_ascii=False, indent=2)
        print("[INFO] 结果写入 {}".format(out_path))
    if can_visualize and n_vis:
        print("[INFO] {} 张叠加图写入 {}".format(n_vis, vis_dir))


def report(args, cfg, action_space, policy, hold, curve, num_samples, forward_seconds,
           inference_timesteps):
    horizon = len(curve)
    print()
    print("=" * 72)
    print("[RESULT] 开环测试 | {} 个样本 x {} 步 chunk | {}".format(
        num_samples, horizon, os.path.basename(args.ckpt)))
    print("         objective={}  action_space={}  ema={}  fp16={}".format(
        cfg.objective, action_space.name, args.ema, cfg.fp16 and not args.fp32))
    print("-" * 72)
    print("{:<22}{:>10}{:>10}{:>10}{:>10}{:>10}".format(
        "metric", "mean", "p50", "p90", "max", "hold"))
    for k in policy:
        print("{:<22}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}".format(
            k, policy[k]["mean"], policy[k]["p50"], policy[k]["p90"],
            policy[k]["max"], hold[k]["mean"]))
    print("-" * 72)
    print("hold = 把当前状态沿整个 chunk 保持不动的基线；它的误差就是 chunk 的真实位移")
    print("量级，策略的 mean 明显小于它才说明学到了动作而不是学到了'别动'。")

    # 沿 chunk 的分布：末端误差远大于均值是正常的，但差得太多说明 chunk 后半段在漂
    curve_keys = [k for k in curve[0] if "last" not in k]
    print("-" * 72)
    print("误差沿 chunk 位置（t=0 是最近的一步）：")
    print("  {:<5}".format("t") + "".join("{:>18}".format(k) for k in curve_keys))
    for t, step in enumerate(curve):
        if horizon > 12 and t not in (0, horizon // 4, horizon // 2,
                                      3 * horizon // 4, horizon - 1):
            continue
        print("  {:<5}".format(t) + "".join("{:>18.4f}".format(step[k])
                                            for k in curve_keys))
    print("-" * 72)
    print("采样耗时 {:.1f}s，{:.3f}s/样本（{} 步去噪）".format(
        forward_seconds, forward_seconds / max(num_samples, 1), inference_timesteps))
    print("=" * 72)


if __name__ == "__main__":
    main()
