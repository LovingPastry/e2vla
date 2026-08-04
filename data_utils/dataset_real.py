"""真机 memmap(.bin) 数据集，对齐 `H5DatasetMapBase.__getitem__` 的输出契约。

数据布局：每条轨迹一个目录，目录下 `metadata.json` 描述若干 `NAME.bin` 的 dtype/shape，
由 `load_dicted_array` 以 np.memmap 打开。需要的字段见 `RealBinDataset.load_traj`。

与 h5 路径的两点差异，都是有意的：
1. 不做时间插值。h5 路径按时间戳插值（`align.interp_SE3_sep`），因为 DROID 这类数据的相机
   与机器人状态不同频。真机 memmap 是等间隔录制的，直接按索引采样即可；`timestamps` 由
   `index * record_dt` 合成，只为满足契约。若你的录制帧率不稳，改回时间戳插值。
2. 不复用 `DataSampler.sample_framedict`。那个函数在 184 行引用了签名里没有的 `skip_rgb`，
   当前是坏的；这里只复用 `preprocess_images` / `pad2ncam` / `pad2nee` 这些可用零件。
"""

import os
import glob
import json
from typing import Dict, List, Optional, Tuple

import numpy as np

from data_utils.dataset_base import (
    DataConfig, DataSampler, H5DatasetMapBase, gen_norm_xy_map,
)
from data_utils.h5io import default_intrinsics, identity_extrinsics


def load_dicted_array(data_path: str, mode: str = "r") -> Dict[str, np.memmap]:
    with open(os.path.join(data_path, "metadata.json"), "r") as fp:
        metadata = json.load(fp)

    data: Dict[str, np.memmap] = {}
    for name, attr in metadata.items():
        data[name] = np.memmap(
            filename=os.path.join(data_path, name + ".bin"),
            mode=mode,
            dtype=attr["dtype"],
            shape=tuple(attr["shape"]),
        )
    return data


def compose_ee_gripper(ee_poses: np.ndarray, openness: np.ndarray) -> np.ndarray:
    """(T, nee, 4, 4) + (T, nee) -> (T, nee, 17)，17 = 行主序展平的 4x4 + 夹爪开合度。

    与 `h5io.compose_ee_gripper` 同构，单独写一份是为了不依赖 h5 那侧的 traj 结构。
    """
    T, nee = ee_poses.shape[:2]
    assert openness.shape[:2] == (T, nee), \
        "openness {} 与 ee_poses {} 的 (T, nee) 不一致".format(openness.shape, ee_poses.shape)
    return np.concatenate([
        ee_poses.reshape(T, nee, 16),
        openness.reshape(T, nee, 1),
    ], axis=-1).astype(np.float32)


class RealBinDataset(H5DatasetMapBase):
    """真机单任务数据集。基类的 `h5_filelist` 这里存的是轨迹目录，不是 h5 文件。

    继承 `H5DatasetMapBase` 不是为了复用 `__getitem__`（整个被覆盖了），而是为了拿到
    `cam_num` / `ee_num` / `pad2ncam` / `pad2nee` / `skip_rgb` 这几个属性——
    `concat_datasets` 和 `data_prepare/compute_action_stats.py` 都依赖它们。
    """

    config = DataConfig(
        sample_dt=1.0,
        # 真机等间隔录制，用它合成 timestamps；填录制周期（秒），例如 30Hz -> 1/30
        record_dt=1.0 / 30,
        output_image_hw=(256, 256),
        ee_indices=(0,),
        # 名字只用于计数 ncam 与 shuffle 顺序；实际取图靠下面的 CAMERA_AXIS
        camera_names=("e2h_cam", "eih_cam"),
        num_history_cameras=1,
        num_history_states=4,
        num_future_states=16,
        sample_camera_gaps=10,
        sample_state_gaps=2,
        shuffle_cameras=False,  # 无真实外参时打乱相机顺序只会制造噪声，见下方 NOTE
    )

    # ---- 需要你按 metadata.json 核对的部分 -------------------------------------
    # rgbs 数组里相机所在的轴顺序，必须与 config.camera_names 对应
    CAMERA_AXIS: Tuple[str, ...] = ("e2h_cam", "eih_cam")
    # 存的是不是 BGR。你 pkl 路径里做过 [:, :, [2,1,0]]，memmap 路径要确认转换是否已在
    # process 阶段做掉；搞反了模型照样收敛，但和预训练权重的色彩统计对不上
    IS_BGR: bool = False
    # 夹爪开合度：优先读独立的 `norm_openness` 数组（已在 [0,1]）；没有则从 joint 的最后
    # 一列按 [GRIPPER_MIN, GRIPPER_MAX] 线性映射到 [0,1]（0=闭合, 1=张开）
    GRIPPER_MIN: float = 0.0
    GRIPPER_MAX: float = 1.5
    PROMPT_TEXT: str = "pick up the red cup and place it in the coffee machine"
    # ---------------------------------------------------------------------------

    def __init__(self, traj_dirs: List[str]):
        super().__init__(traj_dirs)
        assert len(self.CAMERA_AXIS) == len(self.config.camera_names), \
            "CAMERA_AXIS 与 config.camera_names 数量不一致"
        self._cache: Dict[int, Dict[str, np.memmap]] = {}

    @classmethod
    def inst(cls, data_root: str = "/data/lanzc/task0_0716_process/"):
        meta_files = glob.glob(os.path.join(data_root, "**", "metadata.json"), recursive=True)
        traj_dirs = sorted(os.path.dirname(f) for f in meta_files)
        assert len(traj_dirs) > 0, "在 {} 下没找到 metadata.json".format(data_root)
        print("[INFO] num samples of {}: {}".format(cls.__name__, len(traj_dirs)))
        return cls(traj_dirs)

    def load_traj(self, i: int) -> Dict[str, np.memmap]:
        """按 worker 缓存 memmap 句柄。memmap 只映射不读盘，缓存的是 fd 不是数据；
        每次 __getitem__ 重新解析 metadata.json + 重建 memmap 是纯浪费。"""
        if i not in self._cache:
            self._cache[i] = load_dicted_array(self.h5_filelist[i])
        return self._cache[i]

    def get_openness(self, traj: Dict[str, np.memmap]) -> np.ndarray:
        """(L, nee)，[0 (闭合), 1 (张开)]。"""
        if "norm_openness" in traj:
            openness = np.asarray(traj["norm_openness"], dtype=np.float32)
        else:
            # joint 的最后一列是夹爪宽度（物理量），线性归一化
            width = np.asarray(traj["joint"], dtype=np.float32)[:, -1]
            rng = self.GRIPPER_MAX - self.GRIPPER_MIN
            assert rng > 0, "GRIPPER_MAX 必须大于 GRIPPER_MIN"
            openness = np.clip((width - self.GRIPPER_MIN) / rng, 0.0, 1.0)
        if openness.ndim == 1:
            openness = openness[:, None]  # (L,) -> (L, nee=1)
        return openness

    def sample_indices(self, traj_len: int, latest: bool, debug_sample_index: Optional[int]):
        cfg = self.config
        if debug_sample_index is not None:
            last = int(np.clip(debug_sample_index, 0, traj_len - 1))
        elif latest:
            last = traj_len - 1
        else:
            last = int(np.random.choice(traj_len))

        obs_ind = last + np.arange(-cfg.num_history_cameras + 1, 1) * cfg.sample_camera_gaps
        hist_ind = last + np.arange(-cfg.num_history_states + 1, 1) * cfg.sample_state_gaps
        fut_ind = last + (1 + np.arange(cfg.num_future_states)) * cfg.sample_state_gaps

        clip = lambda x: np.clip(x, 0, traj_len - 1)
        return clip(obs_ind), clip(hist_ind), clip(fut_ind)

    def sample_traj(
        self,
        traj: Dict[str, np.memmap],
        latest: bool = False,
        debug_sample_index: Optional[int] = None,
    ):
        cfg = self.config
        all_ee_poses = traj["ee_poses"]           # (L, 4, 4) 或 (L, nee, 4, 4)
        traj_len = len(all_ee_poses)
        obs_ind, hist_ind, fut_ind = self.sample_indices(traj_len, latest, debug_sample_index)

        # ---- 图像 -------------------------------------------------------------
        # memmap 的花式索引会拷贝，这里正是我们要的
        rgbs_raw = np.asarray(traj["rgbs"][obs_ind])  # (To, ncam_raw, 3, H, W)
        assert rgbs_raw.ndim == 5, "rgbs 期望 (L, ncam, 3, H, W)，实得 {}".format(traj["rgbs"].shape)
        assert rgbs_raw.shape[1] == len(self.CAMERA_AXIS), \
            "rgbs 的 ncam={} 与 CAMERA_AXIS={} 不符".format(rgbs_raw.shape[1], self.CAMERA_AXIS)

        if self.IS_BGR:
            rgbs_raw = np.ascontiguousarray(rgbs_raw[:, :, ::-1])

        cam_order = [self.CAMERA_AXIS.index(name) for name in self.camera_order()]
        H, W = rgbs_raw.shape[-2:]
        # NOTE 无标定内参。default_intrinsics 是 D435 的名义 K，对 proj_pe 当位置编码用没问题，
        # 但任何度量用途（投影、跨相机 FOV 比较）都是编的。拿到真实标定后换成逐相机的真值。
        Ks = [default_intrinsics(H, W) for _ in cam_order]
        rgb_list = [rgbs_raw[:, c] for c in cam_order]  # 每个 (To, 3, H, W)

        # preprocess_images 负责 uint8->float32/255、resize，并把 K 跟着 resize 一起变换
        K, rgbs = DataSampler.preprocess_images(
            Ks=Ks, rgbs=rgb_list,
            output_image_hw=None if self.skip_rgb else cfg.output_image_hw,
        )  # K: (ncam, 3, 3); rgbs: (To, ncam, 3, H, W)

        # ---- 位姿与动作 -------------------------------------------------------
        ee_poses = np.asarray(all_ee_poses[obs_ind], dtype=np.float32)
        if ee_poses.ndim == 3:
            ee_poses = ee_poses[:, None]  # (To, 4, 4) -> (To, nee=1, 4, 4)

        openness = self.get_openness(traj)  # (L, nee)

        def gather_actions(ind):
            poses = np.asarray(all_ee_poses[ind], dtype=np.float32)
            if poses.ndim == 3:
                poses = poses[:, None]
            return compose_ee_gripper(poses, openness[ind])  # (T, nee, 17)

        history_actions = gather_actions(hist_ind)
        future_actions = gather_actions(fut_ind)

        # ---- 外参 -------------------------------------------------------------
        # NOTE 无手眼标定。identity 是唯一安全的填充值：ContextEncoder 会把所有相机 rebase
        # 到 cam0 再喂 PRoPE，space_ee2cam 还要对它求逆——零矩阵会在第一次 inverse 出 NaN。
        # 全 identity 时 PRoPE 退化为 no-op，动作从 camera-relative 退化为 base-relative，
        # 仍然自洽。也正因为如此，两路相机在几何上完全不可区分，shuffle_cameras 必须关掉。
        To, ncam = rgbs.shape[:2]
        obs_extrinsics = np.tile(identity_extrinsics(ncam)[None], (To, 1, 1, 1)).astype(np.float32)

        # ---- ee 选取与 padding ------------------------------------------------
        ee_indices = list(self.config.ee_indices)
        assert len(ee_indices) > 0, "ee_indices 不能为空，单臂请填 (0,)"
        ee_poses = ee_poses.take(ee_indices, axis=1)
        history_actions = history_actions.take(ee_indices, axis=1)
        future_actions = future_actions.take(ee_indices, axis=1)

        current_nee = len(ee_indices)
        if self.pad2nee > 0:
            ee_poses = DataSampler.pad2nee(ee_poses, self.pad2nee, dim=1)
            history_actions = DataSampler.pad2nee(history_actions, self.pad2nee, dim=1)
            future_actions = DataSampler.pad2nee(future_actions, self.pad2nee, dim=1)
            valid_ee_mask = np.zeros(self.pad2nee, dtype=bool)
            valid_ee_mask[:current_nee] = True
        else:
            valid_ee_mask = np.ones(current_nee, dtype=bool)

        if self.pad2ncam > 0:
            rgbs = DataSampler.pad2ncam(rgbs, self.pad2ncam, dim=1, zero_init=True)
            obs_extrinsics = DataSampler.pad2ncam(obs_extrinsics, self.pad2ncam, dim=1, zero_init=False)
            K = DataSampler.pad2ncam(K, self.pad2ncam, dim=0, zero_init=False)

        # ---- 组装契约 ---------------------------------------------------------
        H, W = rgbs.shape[-2:]
        norm_xys = gen_norm_xy_map(H, W, K).astype(np.float32)
        norm_xys = norm_xys[None].repeat(To, axis=0)  # (To, ncam, 2, H, W)

        record_dt = cfg.record_dt if cfg.record_dt is not None else cfg.sample_dt
        timestamps = (obs_ind * record_dt).astype(np.float32)  # (To,)

        return {
            "K": K.astype(np.float32),                          # (ncam, 3, 3)
            "rgbs": rgbs.astype(np.float32),                    # (To, ncam, 3, H, W)
            "prompt_text": self.PROMPT_TEXT,                    # str
            "obs_norm_xys": norm_xys,                           # (To, ncam, 2, H, W)
            "obs_extrinsics": obs_extrinsics,                   # (To, ncam, 4, 4)
            "ee_poses": ee_poses[-1],                           # (nee, 4, 4)，仅最新一帧
            "history_actions": history_actions,                 # (nhist, nee, 17)
            "future_actions": future_actions,                   # (Ta, nee, 17)
            "timestamps": timestamps,                           # (To,)
            "valid_ee_mask": valid_ee_mask,                     # (nee,)
        }

    def camera_order(self) -> List[str]:
        names = list(self.config.camera_names)
        if self.config.shuffle_cameras:
            import random
            random.shuffle(names)
        return names

    def __getitem__(self, i):
        return self.sample_traj(self.load_traj(i), latest=False)

    def visualize(self):
        raise NotImplementedError(
            "dataset_base.visualize_dataset 直接 h5py.File(...) 打开文件，不适用于 memmap。"
            "无真实外参时轨迹投影本来也没有意义（visualize_traj 会走 is_identity_extrinsics "
            "分支只显示原图）。"
        )


def check_contract(dataset: RealBinDataset, num_samples: int = 8):
    """按 `H5DatasetMapBase.sample_from_hdf5` 的契约逐键校验形状与取值范围。"""
    cfg = dataset.config
    ncam, nee = dataset.pad2ncam, dataset.pad2nee
    To, nhist, Ta = cfg.num_history_cameras, cfg.num_history_states, cfg.num_future_states
    Hout, Wout = cfg.output_image_hw

    expected = {
        "K": (ncam, 3, 3),
        "rgbs": (To, ncam, 3, Hout, Wout),
        "obs_norm_xys": (To, ncam, 2, Hout, Wout),
        "obs_extrinsics": (To, ncam, 4, 4),
        "ee_poses": (nee, 4, 4),
        "history_actions": (nhist, nee, 17),
        "future_actions": (Ta, nee, 17),
        "timestamps": (To,),
        "valid_ee_mask": (nee,),
    }

    for i in range(min(num_samples, len(dataset))):
        out = dataset[i]
        assert isinstance(out["prompt_text"], str)
        for k, shape in expected.items():
            assert k in out, "缺少键 {}".format(k)
            assert out[k].shape == shape, \
                "{}: 期望 {}, 实得 {}".format(k, shape, out[k].shape)

        assert out["rgbs"].dtype == np.float32 and out["rgbs"].max() <= 1.0, \
            "rgbs 必须是已除 255 的 float32"
        openness = out["future_actions"][..., -1]
        assert openness.min() >= 0.0 and openness.max() <= 1.0, \
            "夹爪开合度必须在 [0,1]，实得 [{:.3f}, {:.3f}]；检查 GRIPPER_MIN/MAX".format(
                openness.min(), openness.max())
        poses = out["future_actions"][..., :16].reshape(Ta, nee, 4, 4)
        assert np.allclose(poses[..., 3, :], [0, 0, 0, 1], atol=1e-4), \
            "位姿末行不是 [0,0,0,1]，ee_poses 可能不是行主序 4x4 齐次矩阵"

    print("[OK] 契约校验通过，{} 个样本".format(min(num_samples, len(dataset))))


def stat_actions(dataset: RealBinDataset, count: int = 1000):
    """打印相邻动作块的平移幅度，用来核对量纲（米 vs 毫米）。
    真正喂给 --action_norm_stats 的统计量请用 data_prepare/compute_action_stats.py，
    那条路径会跑 states2action，量的是模型实际的 10 维动作空间。"""
    deltas = []
    for i in range(min(count, len(dataset))):
        pose = dataset[i]["future_actions"][:, 0, :16].reshape(-1, 4, 4)
        deltas.append(np.abs(pose[-1, :3, 3] - pose[0, :3, 3]))
    deltas = np.stack(deltas, axis=0)
    print("mean:", deltas.mean(axis=0))
    print("std :", deltas.std(axis=0))
    print("max :", deltas.max(axis=0))


if __name__ == "__main__":
    ds = RealBinDataset.inst()
    check_contract(ds)
    stat_actions(ds)
