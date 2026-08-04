"""动作空间：数据集状态 <-> 模型动作 的转换契约。

原来这层是写死的——`ActionExpert.action_dim` 硬返回 10，`states2action` 直接做
`space_ee2cam`，loss 按 [0:3]/[3:9]/[9:10] 切。单任务真机场景下关节角作为输出往往更好用
（不经过 IK，也不受标定误差影响），但那是另一套编码，硬改会把 EE 路径和已发布的预训练
checkpoint 一起弄坏。所以这里把它抽成策略对象，两条路并存：

    CamRelEEPose  —— 原有行为，逐字节等价，`build_action_space("ee_cam")`
    AbsJoint      —— 绝对关节角 + 夹爪，`build_action_space("joint7")`

两个维度必须分清，它们不再相等：
    state_dim   数据集给的宽度（EE: 17 = 展平 4x4 + 夹爪；关节: nq + 1）
    action_dim  模型内部的宽度（EE: 10 = t3r6 + 夹爪；关节: nq + 1）

`layout` 是防串用的标记，同 `train_utils/ckpt.py:OBJECTIVE` 的思路：它会写进统计 json 和
checkpoint 并在加载时校验。这里尤其必要——两套空间的 nq+1 若恰好等于 10，张量形状会完全
一致，权重能干净加载然后输出垃圾，只有这个标记能拦住。
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .action_norm import ActionNormalizer


class ActionSpace(object):
    """数据集状态与模型动作之间的转换。所有方法都是纯函数，不持有参数。"""

    name: str = ""
    layout: str = ""
    state_dim: int = 0
    action_dim: int = 0
    # 是否支持 DiffusionHead 的绝对位置编码（pos_rel2abs）。关节空间要复原末端位置需要
    # 正运动学，得有 URDF/DH 参数，这里没有，所以关掉。
    has_pose_geometry: bool = False

    def states2action(self, cur_wcT: Tensor, cur_weT: Tensor, states: Tensor,
                      action_norm: Optional[ActionNormalizer] = None) -> Tensor:
        raise NotImplementedError

    def action2states(self, cur_wcT: Tensor, cur_weT: Tensor, action: Tensor,
                      action_norm: Optional[ActionNormalizer] = None) -> Tensor:
        raise NotImplementedError

    def init_invalid_states(self, states: Tensor) -> Tensor:
        """给无效 ee 槽位填的占位值，原地写入并返回 `states`。

        EE 空间填单位阵（零矩阵会让下游 `torch.inverse` 出 NaN），关节空间零就够了。
        """
        return states

    def loss(self, pred_noise: Tensor, target: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        raise NotImplementedError


class CamRelEEPose(ActionSpace):
    """原有动作空间：相机相对的 3 平移 + 6D 旋转 + 夹爪开合度。

    行为与重构前逐字节一致，已发布的预训练 checkpoint 走这条路。
    """

    name = "ee_cam"
    layout = "cam_rel_t3r6_openness"
    state_dim = 4 * 4 + 1
    action_dim = 3 + 6 + 1
    has_pose_geometry = True

    def states2action(self, cur_wcT, cur_weT, states, action_norm=None):
        from .action_expert import states2action
        return states2action(cur_wcT, cur_weT, states, action_norm)

    def action2states(self, cur_wcT, cur_weT, action, action_norm=None):
        from .action_expert import action2states
        return action2states(cur_wcT, cur_weT, action, action_norm)

    def init_invalid_states(self, states):
        states[..., :16] = torch.eye(4).ravel().to(states)
        return states

    def loss(self, pred_noise, target):
        pos_loss = F.l1_loss(pred_noise[..., 0:3], target[..., 0:3], reduction="mean")
        rot_loss = F.l1_loss(pred_noise[..., 3:9], target[..., 3:9], reduction="mean")
        openness_loss = F.l1_loss(pred_noise[..., 9:10], target[..., 9:10], reduction="mean")
        total_loss = 30 * pos_loss + 10 * rot_loss + 10 * openness_loss
        return total_loss, {
            "pos_loss": pos_loss.item(),
            "rot_loss": rot_loss.item(),
            "openness_loss": openness_loss.item(),
            "total_loss": total_loss.item(),
        }


class AbsJoint(ActionSpace):
    """绝对关节角 + 夹爪开合度。

    为什么是绝对角而不是增量：这与 EE 路径的设计取向相反，是有意的。EE 那边预测增量是因为
    绝对世界位姿依赖标定，跨 episode 不可比；关节角本身就在机器人自己的坐标里，绝对值天然
    可比，而增量会把误差逐步累积到轨迹末端。ACT / Diffusion Policy 这一系的关节空间实现
    也都是绝对目标。要改成增量的话，覆盖 states2action/action2states 两个方法即可，但记得
    同时改 `layout`，否则旧统计文件会静默套用。

    量纲：这里不做任何归一化，输出的就是弧度 + [-1,1] 的夹爪。缩放交给 ActionNormalizer 的
    q01/q99，和 EE 路径共用同一套机制——用 compute_action_stats 在本空间上重算即可。用关节
    限位 q_min/q_max 归一化也可以，但单任务下 demo 只覆盖限位的一小段，q01/q99 的范围更紧、
    分辨率更高。

    `cur_wcT` / `cur_weT` 在这里用不上，保留在签名里只为与 EE 路径保持同构。
    """

    name = "joint"
    state_dim = 0  # 由 __init__ 按 nq 定
    action_dim = 0
    has_pose_geometry = False

    def __init__(self, num_joints: int = 7):
        assert num_joints > 0
        self.num_joints = num_joints
        self.state_dim = num_joints + 1
        self.action_dim = num_joints + 1
        self.name = "joint{}".format(num_joints)
        self.layout = "abs_joint{}_openness".format(num_joints)

    def states2action(self, cur_wcT, cur_weT, states, action_norm=None):
        """(B, T, nq+1) -> (B, T, nq+1)。夹爪从数据集的 [0,1] 重标定到 [-1,1]，与 EE
        路径的第 2 步一致；关节角原样透传。"""
        assert states.shape[-1] == self.state_dim, \
            "期望 state_dim={}，实得 {}".format(self.state_dim, states.shape[-1])
        joints = states[..., :self.num_joints]
        openness = (states[..., -1:] - 0.5) * 2
        action = torch.cat([joints, openness], dim=-1)
        if action_norm is not None:
            action = action_norm.normalize(action)
        return action

    def action2states(self, cur_wcT, cur_weT, action, action_norm=None):
        if action_norm is not None:
            action = action_norm.unnormalize(action)
        joints = action[..., :self.num_joints]
        openness = action[..., -1:] / 2 + 0.5
        return torch.cat([joints, openness], dim=-1)

    def loss(self, pred_noise, target):
        nq = self.num_joints
        joint_loss = F.l1_loss(pred_noise[..., :nq], target[..., :nq], reduction="mean")
        openness_loss = F.l1_loss(pred_noise[..., nq:nq+1], target[..., nq:nq+1],
                                  reduction="mean")
        # 权重沿用 EC 路径的量级。prediction_type="epsilon" 下两项都是同一个标准正态的切片，
        # 所以这纯粹是 loss 整形，不是量纲补偿。
        total_loss = 30 * joint_loss + 10 * openness_loss
        return total_loss, {
            "joint_loss": joint_loss.item(),
            "openness_loss": openness_loss.item(),
            "total_loss": total_loss.item(),
        }


def build_action_space(spec: Optional[str | ActionSpace]) -> ActionSpace:
    """"ee_cam" | "joint" | "joint7" | "joint6" ... -> ActionSpace。None 取 EE（历史默认）。"""
    if isinstance(spec, ActionSpace):
        return spec
    if spec is None or spec == "":
        return CamRelEEPose()
    spec = spec.strip()
    if spec in ("ee_cam", "ee", "cam_rel_t3r6_openness"):
        return CamRelEEPose()
    if spec == "joint":
        return AbsJoint()
    if spec.startswith("joint"):
        suffix = spec[len("joint"):]
        if suffix.isdigit():
            return AbsJoint(num_joints=int(suffix))
    raise ValueError(
        "未知的 action_space '{}'。可选：'ee_cam'、'joint'（=joint7）、'jointN'".format(spec))
