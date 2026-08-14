"""开环单样本细看：在训练数据上取一个观测，跑完整采样循环，把动作块逐步打出来。

    # 最简：ckpt 目录的 config json 提供数据集、动作空间、归一化、LoRA
    CUDA_VISIBLE_DEVICES=0 python dump_chunk.py --ckpt ./checkpoints/E2VLA/VA_SA/ckpt_latest.pt

    # 换样本 / 连看几个 / 换数据集
    CUDA_VISIBLE_DEVICES=0 python dump_chunk.py --ckpt ... --index 37 -n 3
    CUDA_VISIBLE_DEVICES=0 python dump_chunk.py --ckpt ... --dataset Libero10 --data_root /path

    # 同一个观测重复采 5 次：看采样器的随机性有多大（诊断"塌到均值"用这个）
    CUDA_VISIBLE_DEVICES=0 python dump_chunk.py --ckpt ... --samples 5

    # 顺带打印模型内部的动作向量（t3r6 / 归一化后的值），而不是解码回来的 state
    CUDA_VISIBLE_DEVICES=0 python dump_chunk.py --ckpt ... --raw

与 `test.py` 的分工：那边是遍历数据集出统计量（mean/p50/p90 + hold 基线），回答"这个 ckpt
大概能不能用"；这边只看一个样本的原始数字，回答"它到底预测了什么"。所以这里刻意不做任何
聚合——夹爪卡住、动作块整体不动、chunk 后半段发散这类问题，看统计量只能看出"有问题"，
得把逐步的数打出来才知道是哪一种。

"开环"同 `test.py`：观测全部来自数据集真值，预测不回灌。在训练集上跑时它先答的是拟合，
不是泛化。

模型构建与权重加载整个交给 `infer_utils.planner.load_model`（LoRA 注入顺序、EMA copy_to、
objective / action_layout / action_norm 三个戳的校验都在那里），数据侧复用 `test.py` 的
`FixedSampleView` / `resolve_dataset_classes` / `instantiate`：两个脚本必须在同一份数据和
同一套 padding 上，否则两边的数字没法互相印证。
"""

import os
import argparse
from typing import Dict, List, Tuple

import envars  # noqa: F401  必须在 torchvision（被 data_utils 带进来）之前 import
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset

# 同目录的 test.py（不是标准库的 test 包——脚本目录在 sys.path[0]）。复用它的数据装配，
# 见模块 docstring 最后一段。
from test import FixedSampleView, resolve_dataset_classes, instantiate, flatten_valid
from models.action_norm import EE_POSE_LAYOUTS
from models.action_space import ActionSpace, reference_cam_pose
from infer_utils.planner import load_model, parse_config


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------

def build_view(cfg, args):
    """按 ckpt 的训练配置搭出一个可按序号索引的数据视图。

    与 `test.py:build_dataloader` 同源，两处差异都是"只看一个样本"带来的：不建 DataLoader
    （直接 `view[k]`，省掉为一个样本起 worker），也不截断成 Subset（`--index` 要能指到任何
    位置）。padding 与 shuffle_cameras 的处理必须保持一致，否则输入宽度和相机顺序都变了。
    """
    classes = resolve_dataset_classes(cfg, args.dataset)
    print("[INFO] 数据集: {}".format(", ".join(D.__name__ for D in classes)))

    # 相机顺序在部署时不打乱（`planner.parse_config` 也这么干）：camera_names[0] 定义了
    # 整个动作坐标系。要复现的是部署条件，不是训练时的数据增广。
    for D in classes:
        if D.config.shuffle_cameras:
            print("[INFO] {}: shuffle_cameras True -> False（对齐部署条件）"
                  .format(D.__name__))
            D.config.shuffle_cameras = False

    ds_list = [instantiate(D, args.data_root) for D in classes]

    # padding 对齐**训练时**所有数据集的最大相机 / ee 数，而不是本次子集的：模型见过的
    # 就是那个宽度（多出来的相机是零填充）。按子集重算既不报错也不缺 key，只会静默换掉
    # 输入宽度。
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
    print("[INFO] {} 条 episode x {} 次采样 = {} 个可选样本".format(
        len(dataset), args.repeats, len(view)))
    return view, ds_list


def collate_one(sample: Dict, device: str) -> Tuple[Dict, Dict]:
    """单个样本 -> batch size 为 1 的模型输入。

    手写而不用 `default_collate`：要的只是加一维，而 collate 会把 `prompt_text` 这类
    非张量字段按类型分派，行为随 torch 版本变。
    """
    batch = {}
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            batch[k] = torch.from_numpy(v).unsqueeze(0).to(device)
        elif isinstance(v, Tensor):
            batch[k] = v.unsqueeze(0).to(device)
        else:
            batch[k] = [v]  # prompt_text: str -> [str]，模型按 List[str] 取
    return batch


# ---------------------------------------------------------------------------
# state -> 人能读的列
# ---------------------------------------------------------------------------

def matrix_to_rpy_deg(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 3) 的 roll/pitch/yaw（度，ZYX 外旋 = XYZ 内旋约定）。

    只用于打印。万向节奇异点（|pitch| = 90°）附近 roll/yaw 会互相搬运数值，所以判断"姿态
    有没有在动"要看下面的 `drot_deg`（相对当前姿态的转角，无奇异），别看这三个数。
    """
    sy = -R[..., 2, 0]
    cy = np.sqrt(np.clip(1.0 - sy ** 2, 0.0, 1.0))
    pitch = np.arcsin(np.clip(sy, -1.0, 1.0))
    # cy≈0 时 roll 与 yaw 退化成同一个自由度，按惯例把它全给 roll
    degenerate = cy < 1e-6
    roll = np.where(degenerate,
                    np.arctan2(-R[..., 1, 2], R[..., 1, 1]),
                    np.arctan2(R[..., 2, 1], R[..., 2, 2]))
    yaw = np.where(degenerate, 0.0, np.arctan2(R[..., 1, 0], R[..., 0, 0]))
    return np.rad2deg(np.stack([roll, pitch, yaw], axis=-1))


def rot_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> np.ndarray:
    """两个旋转之间的转角（度）。trace(R) = 1 + 2cos(theta)，与
    `ActionSpace.state_error` 用的是同一个式子和同一处 clamp——网络输出不保证正交，
    不 clamp 的话 arccos 直接给 NaN。"""
    rel = np.swapaxes(R_a, -1, -2) @ R_b
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    return np.rad2deg(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))


def decode_columns(action_space: ActionSpace, states: np.ndarray,
                   cur_state: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """(T, state_dim) -> (列名, (T, C)) 的可读表格。

    EE 空间：世界系位置（米）+ rpy（度）+ 相对当前位姿的位移/转角 + 夹爪。
    关节空间：绝对关节角（弧度）+ 夹爪。
    夹爪固定放最后一列，`gripper_column` 依赖这一点。
    """
    if action_space.layout in EE_POSE_LAYOUTS:
        T = states.shape[0]
        pose = states[:, :16].reshape(T, 4, 4)
        cur_pose = cur_state[:16].reshape(4, 4)
        pos, R = pose[:, :3, 3], pose[:, :3, :3]
        rpy = matrix_to_rpy_deg(R)
        # 相对当前位姿的量：绝对 xyz 看不出"chunk 到底走了多远"，而这正是判断动作块
        # 有没有在动的那个数（也是 test.py 里 hold 基线在量的东西）
        dpos_mm = np.linalg.norm(pos - cur_pose[:3, 3], axis=-1) * 1000.0
        drot_deg = rot_angle_deg(np.broadcast_to(cur_pose[:3, :3], R.shape), R)
        cols = ["x/m", "y/m", "z/m", "roll", "pitch", "yaw", "|dp|mm", "|dR|deg", "grip"]
        table = np.concatenate([pos, rpy, dpos_mm[:, None], drot_deg[:, None],
                                states[:, -1:]], axis=-1)
        return cols, table

    nq = states.shape[-1] - 1
    cols = ["q{}/rad".format(i) for i in range(nq)] + ["grip"]
    return cols, states.copy()


def gripper_column(states: np.ndarray) -> np.ndarray:
    """夹爪那一列，[0 (闭合), 1 (张开)]。两个动作空间都把它放在 state 的最后一维。"""
    return states[..., -1]


# ---------------------------------------------------------------------------
# 打印
# ---------------------------------------------------------------------------

def print_table(cols: List[str], gt: np.ndarray, preds: List[np.ndarray], stride: int):
    """真值与各次采样交错成一张表：每个 t 一组，gt 一行、pd* 各一行。

    交错而不是左右并排：并排要 2C 列，超过终端宽度就自动折行，反而更难读；上下相邻的
    同名列可以直接竖着比。
    """
    width = 9
    header = "  {:<4}{:<5}".format("t", "") + "".join(
        "{:>{w}}".format(c, w=width) for c in cols)
    print(header)
    print("  " + "-" * (len(header) - 2))

    T = gt.shape[0]
    steps = [t for t in range(T) if t % stride == 0]
    if steps and steps[-1] != T - 1:
        steps.append(T - 1)  # 末端永远要看到：误差沿 chunk 累积

    def row(t_label: str, tag: str, values: np.ndarray):
        print("  {:<4}{:<5}".format(t_label, tag) + "".join(
            "{:>{w}.3f}".format(v, w=width) for v in values))

    for t in steps:
        row(str(t), "gt", gt[t])
        for i, pred in enumerate(preds):
            row("", "pd" if len(preds) == 1 else "pd{}".format(i + 1), pred[t])
        print()


def print_gripper_summary(gt: np.ndarray, preds: List[np.ndarray], cur_grip: float,
                          threshold: float = 0.5):
    """夹爪单独再看一眼：它是唯一离散跳变的通道，塌到均值时表格里不明显，这里明显。

    `变化次数` 数的是二值化后的翻转次数，也就是下游真正会执行的那个量——
    `examples/libero/eval.py` 就是 `(g > 0.5)` 之后再映射成 -1/+1 的指令。
    """
    def describe(g: np.ndarray) -> str:
        binary = g > threshold
        flips = int(np.count_nonzero(binary[1:] != binary[:-1]))
        return ("min {:.3f}  max {:.3f}  极差 {:.3f}  过阈值翻转 {} 次  "
                "二值序列 {}".format(g.min(), g.max(), g.max() - g.min(), flips,
                                     "".join("1" if b else "0" for b in binary)))

    print("  [夹爪] 当前状态（history 最后一帧）= {:.3f}".format(cur_grip))
    print("  {:<8}{}".format("gt", describe(gripper_column(gt))))
    for i, pred in enumerate(preds):
        tag = "pd" if len(preds) == 1 else "pd{}".format(i + 1)
        print("  {:<8}{}".format(tag, describe(gripper_column(pred))))

    if len(preds) > 1:
        stacked = np.stack([gripper_column(p) for p in preds])  # (K, T)
        print("  {:<8}逐步标准差 max {:.4f}（多次采样之间的差异；≈0 说明采样器没有随机性，"
              "头已经塌成确定映射）".format("spread", stacked.std(axis=0).max()))

    # np.ptp(...) 而不是 ndarray.ptp()：后者在 NumPy 2.0 里被移除了
    gt_range = float(np.ptp(gripper_column(gt)))
    pred_range = max(float(np.ptp(gripper_column(p))) for p in preds)
    if pred_range < 0.1 and gt_range > 0.3:
        print("  [WARN] 真值夹爪在动（极差 {:.3f}）而预测基本是常量（极差 {:.3f}）——"
              "这就是'夹爪不动'".format(gt_range, pred_range))


def print_raw(action_space: ActionSpace, action_norm, cam_pose: Tensor, ee_pose: Tensor,
              gt_states: Tensor, pred_states: Tensor, stride: int):
    """模型内部那一层：state -> action（t3r6 / 绝对关节角，若开了归一化则是归一化后的值）。

    表格里的数是解码回 state 之后的，看不出编码本身有没有问题。这里走的是与训练时逐字节
    相同的 `states2action`，所以打出来的就是 loss 真正作用在的那个向量——归一化范围是否
    合理（|值| 应大致落在 1 以内）、6D 旋转的两个 3-向量是否退化，都在这层才看得见。
    """
    def encode(states: Tensor) -> np.ndarray:
        act = action_space.states2action(cam_pose, ee_pose, states.unsqueeze(0),
                                        action_norm)
        return act[0].float().cpu().numpy()

    gt_act, pred_act = encode(gt_states), encode(pred_states)
    dim = gt_act.shape[-1]
    if action_space.layout in EE_POSE_LAYOUTS:
        cols = ["t{}".format(i) for i in range(3)] + \
               ["r{}".format(i) for i in range(6)] + ["grip"]
    else:
        cols = ["q{}".format(i) for i in range(dim - 1)] + ["grip"]

    print("  内部动作向量（{}{}）：".format(
        action_space.layout, "，已归一化" if action_norm is not None else "，未归一化"))
    print_table(cols, gt_act, [pred_act], stride)


def print_sample(action_space: ActionSpace, sample: Dict, gt: np.ndarray,
                 preds: List[np.ndarray], cur_state: np.ndarray, ee_slot: int,
                 stride: int):
    cols, gt_table = decode_columns(action_space, gt, cur_state)
    pred_tables = [decode_columns(action_space, p, cur_state)[1] for p in preds]

    print("-" * 78)
    print("  样本 {} | episode {} | ee 槽位 {} | prompt: {}".format(
        int(sample["eval_index"]), int(sample["episode_index"]), ee_slot,
        sample.get("prompt_text", "")))
    print("-" * 78)
    print_table(cols, gt_table, pred_tables, stride)
    print_gripper_summary(gt, preds, float(gripper_column(cur_state)))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="开环单样本细看：把一组 action chunk 的预测与真值逐步打出来",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ckpt", type=str, required=True,
                        help="ckpt 路径；同目录下的 *.json 提供模型与数据配置")
    parser.add_argument("--dataset", type=str, nargs="+", default=None,
                        help="数据集类名，默认用 ckpt config 里记的（python datavis.py -l 列全部）")
    parser.add_argument("--data_root", type=str, default=None,
                        help="仅对 inst() 接受 data_root 的数据集有效（如 RealBinDataset）")
    parser.add_argument("--index", type=int, default=0, help="从第几个样本开始看")
    parser.add_argument("-n", "--num_chunks", type=int, default=1, help="连着看几个样本")
    parser.add_argument("--samples", type=int, default=1,
                        help="同一个观测重复采样几次（>1 用来看采样器的随机性）")
    parser.add_argument("--stride", type=int, default=1, help="chunk 内每隔几步打一行")
    parser.add_argument("--raw", action="store_true",
                        help="额外打印模型内部的动作向量（t3r6 / 归一化后的值）")
    parser.add_argument("--repeats", type=int, default=1,
                        help="每条 episode 在不同时刻采几次；决定 --index 的编号空间")
    parser.add_argument("--seed", type=int, default=0,
                        help="决定采到哪些时刻；要和 test.py 对得上就保持一致")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ema", action="store_true",
                        help="用 EMA 影子权重，仅当训练时真开了 ema_enabled")
    parser.add_argument("--fp32", action="store_true",
                        help="关掉 bfloat16，默认跟随 ckpt config 的 fp16")
    parser.add_argument("--pad_ncam", type=int, default=0, help="0 表示按训练配置推断")
    parser.add_argument("--pad_nee", type=int, default=0, help="0 表示按训练配置推断")
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    assert args.samples >= 1 and args.num_chunks >= 1 and args.stride >= 1

    cfg, _ = parse_config(os.path.dirname(os.path.abspath(args.ckpt)))
    # 权重加载的全部规矩都在 load_model 里，包括三个戳的校验
    model, _ = load_model(args.ckpt, args.device, use_ema=args.ema)
    action_space = model.action_space
    action_norm = model.actor.action_norm
    fp16 = cfg.fp16 and not args.fp32

    print("=" * 78)
    print("[INFO] ckpt {}".format(os.path.basename(args.ckpt)))
    print("[INFO] objective={}  action_space={}  layout={}".format(
        cfg.objective, action_space.name, action_space.layout))
    print("[INFO] context_encoder={}  action_norm={}  采样步数={}  fp16={}  ema={}".format(
        cfg.context_encoder, "开" if action_norm is not None else "关",
        model.actor.inference_timesteps, fp16, args.ema))

    view, _ = build_view(cfg, args)
    last = args.index + args.num_chunks
    assert args.index >= 0 and last <= len(view), \
        "--index/-n 越界：可选样本 [0, {})".format(len(view))

    for k in range(args.index, last):
        sample = view[k]
        batch = collate_one(sample, args.device)

        # 每次 forward 都重新抽噪声，所以循环 --samples 次就是同一观测的多次采样
        preds = []
        for _ in range(args.samples):
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
            )  # (1, Ta, Nee, state_dim)
            preds.append(pred)

        mask = batch["valid_ee_mask"]
        # 与模型内部一致地展平 (B, Nee) -> B'：无效槽位装的是占位填充（EE 是单位阵）
        gt_flat = flatten_valid(batch["future_actions"], mask).float()
        pred_flats = [flatten_valid(p, mask).float() for p in preds]
        if gt_flat.shape[0] == 0:
            print("[WARN] 样本 {} 没有有效 ee，跳过".format(k))
            continue
        for p in pred_flats:
            if not torch.isfinite(p).all():
                print("[WARN] 样本 {} 的预测里有 NaN/Inf".format(k))
                break

        hist_flat = flatten_valid(batch["history_actions"], mask).float()
        valid_slots = torch.nonzero(mask[0]).ravel().tolist()

        for b in range(gt_flat.shape[0]):
            print_sample(
                action_space=action_space,
                sample=sample,
                gt=gt_flat[b].cpu().numpy(),
                preds=[p[b].cpu().numpy() for p in pred_flats],
                cur_state=hist_flat[b, -1].cpu().numpy(),
                ee_slot=valid_slots[b] if b < len(valid_slots) else b,
                stride=args.stride,
            )

            if args.raw:
                # 动作编码所用的参考系，与 `ActionExpert.forward` 走同一个函数：ee_cam 下
                # 是最新帧 0 号相机，无相机参数的模型下是单位阵（= 基座系）
                extrinsics = (batch["obs_extrinsics"]
                              if model.uses_camera_params else None)
                cam_pose = reference_cam_pose(action_space, extrinsics,
                                              batch_size=1, like=batch["ee_poses"])
                ee_pose = batch["ee_poses"][mask]  # (B', 4, 4)
                # cam_pose 是 (1, 4, 4)：batch 只有一个元素，同 batch 内各 ee 共用参考系
                print_raw(action_space, action_norm, cam_pose, ee_pose[b:b + 1],
                          gt_flat[b], pred_flats[0][b], args.stride)

    print("=" * 78)
    print("怎么读：`gt` 是数据集真值，`pd` 是采样出来的预测，两行竖着比。")
    print("  * 位姿/关节：看 |dp|mm 与 |dR|deg（EE）或 q* 的变化量——预测的变化量远小于")
    print("    真值，说明学到的是'别动'而不是动作（等价于 test.py 里输给 hold 基线）。")
    print("  * 夹爪：看 [夹爪] 那几行的极差与翻转次数。真值有翻转、预测极差 < 0.1，")
    print("    就是夹爪塌成常量；--samples 5 时 spread ≈ 0 说明连采样随机性都没了。")
    print("=" * 78)


if __name__ == "__main__":
    main()
