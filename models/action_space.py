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

`layout` 是防串用的标记，同 `train_utils/ckpt.py:check_objective` 的思路：它会写进统计 json 和
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

    def loss(self, pred: Tensor, target: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """按通道切分的加权 L1。

        `pred`/`target` 的含义由 objective 决定——DDIM 下是噪声，flow 下是速度场——但切
        法只取决于动作编码，所以两种目标共用这一个实现。注意权重的性质会跟着变：epsilon
        目标的每一段都是同一个标准正态的切片，权重纯粹是 loss 整形；flow 的目标
        `actions - noise` 带着动作空间自己的量纲，同一组权重就顺带成了量纲补偿。两种
        目标的 loss 数值因此不可比。
        """
        raise NotImplementedError

    def state_error(self, pred_states: Tensor, gt_states: Tensor) -> Dict[str, float]:
        """采样出的动作与真值之间的误差，单位是物理量（米 / 度 / 弧度）。

        这和 `loss` 不是一回事，也不能互相替代。`loss` 衡量的是单步去噪回归得准不准，
        量纲被 objective 和权重搅在一起，跨 objective 甚至不可比；这里衡量的是**跑完整个
        采样循环之后**的动作误差，与 rollout 成功率同量纲、跨 objective 可比，也是唯一
        一个不看仿真就能读出"这个 checkpoint 大概能不能用"的数。

        代价是要跑一遍完整推理，所以 `train.py` 用 `log_sample_interval` 单独控制频率，
        默认关闭。输入是两侧的 state（数据集那一侧的宽度 `state_dim`），不是 action：
        误差要在归一化和相机相对编码都还原之后才有物理意义。

        Args:
            pred_states / gt_states: (B, Ta, state_dim)
        """
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

    def loss(self, pred, target):
        pos_loss = F.l1_loss(pred[..., 0:3], target[..., 0:3], reduction="mean")
        rot_loss = F.l1_loss(pred[..., 3:9], target[..., 3:9], reduction="mean")
        openness_loss = F.l1_loss(pred[..., 9:10], target[..., 9:10], reduction="mean")
        total_loss = 30 * pos_loss + 10 * rot_loss + 10 * openness_loss
        return total_loss, {
            "pos_loss": pos_loss.item(),
            "rot_loss": rot_loss.item(),
            "openness_loss": openness_loss.item(),
            "total_loss": total_loss.item(),
        }

    @torch.no_grad()
    def state_error(self, pred_states, gt_states):
        pred_T = pred_states[..., :16].reshape(*pred_states.shape[:-1], 4, 4)
        gt_T = gt_states[..., :16].reshape(*gt_states.shape[:-1], 4, 4)

        pos_err = (pred_T[..., :3, 3] - gt_T[..., :3, 3]).norm(dim=-1)  # (B, Ta) 米

        # 相对旋转的转角：trace(R) = 1 + 2cos(theta)。clamp 是必须的——数值上 trace 会
        # 越界一点点，acos 直接给 NaN，而这里两个矩阵都来自网络输出，不保证正交。
        # 边界取严格的 ±1 而不是 ±(1-eps)：后者会给完全正确的预测留下一个 acos(1-eps)
        # 的误差地板（1e-6 就是 0.08°），而 acos 在 ±1 上本身是良定义的。
        rel_R = pred_T[..., :3, :3].transpose(-1, -2) @ gt_T[..., :3, :3]
        trace = rel_R[..., 0, 0] + rel_R[..., 1, 1] + rel_R[..., 2, 2]
        cos_theta = ((trace - 1) / 2).clamp(-1.0, 1.0)
        rot_err = torch.rad2deg(torch.acos(cos_theta))  # (B, Ta) 度

        grip_err = (pred_states[..., -1] - gt_states[..., -1]).abs()
        grip_acc = ((pred_states[..., -1] > 0.5) == (gt_states[..., -1] > 0.5)).float()

        return {
            "pos_err_m": pos_err.mean().item(),
            # chunk 末端单独看一眼：误差沿 chunk 累积，平均值会把它摊平，而真正决定
            # 闭环表现的恰恰是最后几步（执行到那里时下一次重规划还没发生）。
            "pos_err_last_m": pos_err[..., -1].mean().item(),
            "rot_err_deg": rot_err.mean().item(),
            "rot_err_last_deg": rot_err[..., -1].mean().item(),
            "grip_l1": grip_err.mean().item(),
            "grip_acc": grip_acc.mean().item(),
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

    def loss(self, pred, target):
        nq = self.num_joints
        joint_loss = F.l1_loss(pred[..., :nq], target[..., :nq], reduction="mean")
        openness_loss = F.l1_loss(pred[..., nq:nq+1], target[..., nq:nq+1],
                                  reduction="mean")
        # 权重沿用 EE 路径的量级，性质见基类 `ActionSpace.loss` 的说明。
        total_loss = 30 * joint_loss + 10 * openness_loss
        return total_loss, {
            "joint_loss": joint_loss.item(),
            "openness_loss": openness_loss.item(),
            "total_loss": total_loss.item(),
        }

    @torch.no_grad()
    def state_error(self, pred_states, gt_states):
        nq = self.num_joints
        joint_err = (pred_states[..., :nq] - gt_states[..., :nq]).abs()  # (B, Ta, nq) 弧度
        grip_err = (pred_states[..., -1] - gt_states[..., -1]).abs()
        grip_acc = ((pred_states[..., -1] > 0.5) == (gt_states[..., -1] > 0.5)).float()
        return {
            "joint_err_rad": joint_err.mean().item(),
            "joint_err_deg": torch.rad2deg(joint_err.mean()).item(),
            # 单个关节的最大误差：平均值会被 7 个关节摊平，而一个腕关节偏 0.3 rad 就足够
            # 让抓取失败。
            "joint_err_max_rad": joint_err.amax(dim=-1).mean().item(),
            "joint_err_last_rad": joint_err[:, -1].mean().item(),
            "grip_l1": grip_err.mean().item(),
            "grip_acc": grip_acc.mean().item(),
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
