"""StateEncoder / ActionDecoder 示例 —— 完全沿用 e2vla 的架构与位置编码写法。

把本文件放到 `e2vla/models/` 下，下面这些层都是它现成的：
    layers/pe.py       SinusoidalPosEmb（加性，序列 index） / RoPE（乘性，任意 position_dim）
                       / PRoPE（乘性，相机位姿）
    layers/utils.py    simple_mlp
    layers/attn_dn.py  SelfAttentionLayer / CrossAttentionLayer / FFN
                       / FFWSelfAttentionLayers / init_xncoder（DeepNorm 初始化）
    dit.py             DiTBlock = self-attn -> cross-attn -> FFN，adaLN 由 film 驱动
    qformer.py         QFormerITM

与 e2vla 原版的三处差异（这里是单任务 pixels->action，没有语言、没有 VLA 泛化）：
  1. 语言链整条删掉：proj_l / x_l / lang_mask / QFormerITM 里的 text 分支
  2. mask 全删：定长动作 chunk + 固定相机数，用不到 concat_mask 那套
  3. PRoPE 要相机外参，MetaWorld 这边拿不到 —— 换成 RoPE(position_dim=3) 编码 (t, y, x)，
     还是它同一套乘性 PE 机器，只是位置量从 4x4 位姿换成时空索引

位置编码分工（和 e2vla ContextEncoder 一致，三件套各管一层语义）：
    加性 proj_pe(归一化 xy)   -> 绝对空间位置，zero-init 所以训练初期不扰动预训练特征
    加性 SinusoidalPosEmb(t)  -> 历史帧序号 / 动作 chunk 序号
    乘性 RoPE(t, y, x)        -> 注意力内部的相对时空位置
"""

from typing import Optional, Sequence

import torch
from torch import nn, Tensor
from einops import rearrange

from .layers.pe import SinusoidalPosEmb, RoPE
from .layers.utils import simple_mlp
from .layers.attn_dn import (
    CrossAttentionLayer,
    FFN,
    FFWSelfAttentionLayers,
    FFWCrossAttentionLayers,
    init_xncoder,
)

# Diffusion Policy 的 FiLM 残差块（ConditionalUnet1D 里那个）
from .conditional_unet1d import ConditionalResidualBlock1D

# Mamba U-Net：把 pvrobo/src/agent/unet_mamba.py 整个拷过来即可，
# 它只依赖 torch，装了 mamba_ssm 用官方 CUDA kernel，没装则走纯 PyTorch 回退
from .unet_mamba import MambaUNet1D


class StateEncoder(nn.Module):
    """历史 H 帧 x V 相机的 DINOv2 patch token + 历史动作 -> context。

    拓扑照 e2vla 的 ContextEncoder：投影 -> 加 PE -> pre_attn 自注意力
    -> query 瓶颈（原版 QFormer 去掉 text 分支就是这个）-> post_attn 自注意力。

    query 瓶颈不是可选项：H=2, V=2, grid=16 时 token 数就是 2*2*256=1024，
    直接丢给 50 步的 decoder 做交叉注意力太贵，压到 num_queries 个再往下传。
    """

    def __init__(
        self,
        vis_dim: int = 384,      # DINOv2 patch token 维度（vit_small = 384）
        action_dim: int = 4,
        hdim: int = 384,
        num_heads: int = 8,
        num_layers: int = 4,     # 前一半 pre_attn，后一半 post_attn
        num_queries: int = 64,
        num_cams: int = 2,
        grid: int = 16,          # 224 / patch14 = 16，patch 数 P = grid^2
    ):
        super().__init__()
        # RoPE 把 head_dim 均分给 position_dim 个轴，(t,y,x) 三轴 -> head_dim 必须能被 3 整除
        assert hdim % (3 * num_heads) == 0, (
            f"hdim={hdim}, num_heads={num_heads}: RoPE(position_dim=3) 要求 "
            f"hdim % (3*num_heads) == 0，例如 hdim=384/num_heads=8 (head_dim=48)"
        )
        self.grid = grid

        # e2vla 有 proj_v[0]/proj_v[1] 分别吃 dinov2/siglip 再相加；这里只有一个
        # backbone，所以一条投影就够
        self.proj_v = simple_mlp([vis_dim, hdim, hdim], ln=True)
        self.proj_a = simple_mlp([action_dim, hdim, hdim], ln=True)
        self.proj_pe = simple_mlp([2, hdim, hdim], ln=True)

        self.cam_embed = nn.Parameter(torch.zeros(num_cams, hdim))
        # 视觉 token 和动作 token 混在同一条序列里，必须能被区分开
        self.role_embed = nn.Parameter(torch.zeros(2, hdim))
        self.time_embed = SinusoidalPosEmb(hdim)
        self.rope = RoPE(hdim, position_dim=3, num_heads=num_heads)

        n_pre = num_layers // 2
        self.pre_attn = FFWSelfAttentionLayers(
            hdim, num_heads, n_pre,
            use_adaln=False, bias=True, qk_norm=True, ffn_expansion=2,
        )
        self.queries = nn.Parameter(torch.randn(1, num_queries, hdim))
        self.bottleneck_attn = CrossAttentionLayer(hdim, num_heads, bias=True, qk_norm=True)
        self.bottleneck_ffn = FFN(hdim, 2 * hdim)
        self.post_attn = FFWSelfAttentionLayers(
            hdim, num_heads, num_layers - n_pre,
            use_adaln=False, bias=True, qk_norm=True, ffn_expansion=2,
        )

        # patch 网格坐标：整数索引喂 RoPE（[0,grid) 的量程正好覆盖若干个周期），
        # 归一化到 [-1,1] 的喂加性 proj_pe —— 对应 e2vla 的 norm_xy_ds
        ys, xs = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
        grid_idx = torch.stack([ys, xs], dim=-1).reshape(-1, 2).float()   # (P, 2)
        self.register_buffer("grid_idx", grid_idx, persistent=False)
        self.register_buffer("grid_norm", grid_idx / (grid - 1) * 2 - 1, persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.queries, std=0.02)
        # zero-init 空间 PE 的输出层：训练初期 x_v 就是干净的 DINOv2 特征。
        # weight 和 bias 都要清零 —— 只清 weight 的话 proj_pe 退化成一个随机常量偏移，
        # 照样扰动预训练特征，zero-init 等于白做。
        nn.init.zeros_(self.proj_pe[-1].weight)
        nn.init.zeros_(self.proj_pe[-1].bias)
        init_xncoder(self.pre_attn.num_layers, self.pre_attn)
        init_xncoder(self.post_attn.num_layers, self.post_attn)

    def _rope_code(self, B: int, H: int, V: int, P: int, device, dtype) -> Tensor:
        """给 (H*V*P 个视觉 token + H 个动作 token) 造 (t, y, x) 坐标 -> RoPE code。"""
        t_vis = (
            torch.arange(H, device=device, dtype=torch.float32)
            .view(H, 1, 1, 1).expand(H, V, P, 1)
        )
        yx_vis = self.grid_idx.view(1, 1, P, 2).expand(H, V, P, 2)
        vis_pos = torch.cat([t_vis, yx_vis], dim=-1).reshape(H * V * P, 3)

        # 动作 token 没有空间坐标 -> (t, 0, 0)，即只在时间轴上旋转
        t_act = torch.arange(H, device=device, dtype=torch.float32).view(H, 1)
        act_pos = torch.cat([t_act, torch.zeros(H, 2, device=device)], dim=-1)

        pos = torch.cat([vis_pos, act_pos], dim=0)[None].expand(B, -1, -1)
        return self.rope(pos).to(dtype)     # (B, L, head_dim, 2)

    def forward(self, tokens: Tensor, actions: Tensor) -> Tensor:
        """
        Args:
            tokens:  (B, H, V, P, vis_dim)  DINOv2 patch token，来自 DinoAFA.encode
            actions: (B, H, action_dim)     历史动作

        Returns:
            context: (B, num_queries, hdim)
        """
        B, H, V, P, _ = tokens.shape
        t_pe = self.time_embed(torch.arange(H, device=tokens.device).to(tokens))  # (H, hdim)

        x = self.proj_v(tokens)                          # (B,H,V,P,hdim)
        x = x + self.proj_pe(self.grid_norm)             # 空间（加性），广播 (P,hdim)
        x = x + self.cam_embed[None, None, :, None]      # 相机身份
        x = x + t_pe[None, :, None, None]                # 时间（加性）
        x = x + self.role_embed[0]
        x = x.reshape(B, H * V * P, -1)

        a = self.proj_a(actions) + t_pe[None] + self.role_embed[1]   # (B,H,hdim)
        x = torch.cat([x, a], dim=1)                                 # (B, H*V*P+H, hdim)

        x_pe = self._rope_code(B, H, V, P, x.device, x.dtype)
        x = self.pre_attn(query=x, query_pe=x_pe)[-1]

        q = self.queries.expand(B, -1, -1)
        q, _ = self.bottleneck_attn(query=q, value=x)
        q = self.bottleneck_ffn(q)
        return self.post_attn(query=q)[-1]
    
# 动作向量布局（action_dim = 8）：
#     x[..., 0:3]   末端位置 xyz
#     x[..., 3:7]   姿态（4 维）
#     x[..., 7:8]   夹爪开合
# 前 3 维有双重身份：既是要预测的量，又被当作 3D 空间坐标喂给 rope3d / xyz_proj。
# 也就是说位置编码取自**含噪动作自身的坐标**，不是 chunk 内的序号
# （3D Diffuser Actor 那条路线）。


class ActionDecoder(nn.Module):
    """含噪动作 chunk + context -> 去噪目标（eps 或速度场，取决于训练目标）。

    命名按 diffusion / flow matching 惯例：
        x       含噪动作序列 (B, Ta, action_dim)
        t       去噪时刻 (B,)
        t_emb   去噪时刻的嵌入，驱动 adaLN，是全局条件
        h       网络内部隐特征 (B, L, hdim)

    这里有两个都叫"时间"的东西，务必分清：
        t          去噪步（diffusion / flow 的积分变量）      -> t_emb -> adaLN film
        chunk 序号  动作序列里的第几步                        -> seq_pos_emb（加性）
    原版的 denoising_time_embed / traj_time_embed 就是这一对，改名成
    t_embed / seq_pos_emb 以后不会再看混。

    还有两套空间位置编码，同源不同用法：
        rope3d(xyz)    乘性，只在注意力内部作用于 q/k，编码相对位置
        xyz_proj(xyz)  加性，直接加到特征上，编码绝对位置

    数据流：
        x -> x_proj + seq_pos_emb                动作 token
        [动作 token ; context] 拼接 -> 联合自注意力（PE 同样拼接）
        切回动作段 -> 交叉注意力读 [自身 ; context]
        -> 1D 卷积补局部时序 -> 双头输出
    """

    def __init__(self, hdim: int, num_heads: int, action_dim: int = 8):
        super().__init__()
        self.action_dim = action_dim

        # ---------------- 输入投影 ----------------
        self.x_proj = nn.Sequential(          # 含噪动作 -> hdim
            nn.Linear(action_dim, hdim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hdim, hdim),
        )
        self.xyz_proj = nn.Sequential(        # 末端 xyz -> hdim，加性空间 PE
            nn.Linear(3, hdim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hdim, hdim),
        )
        self.rope3d = RoPE(hdim, position_dim=3, num_heads=num_heads)  # 乘性空间 PE

        # ---------------- 两种"时间" ----------------
        self.t_embed = nn.Sequential(         # 去噪时刻 -> adaLN 的 film
            SinusoidalPosEmb(hdim),
            nn.Linear(hdim, hdim),
            nn.ReLU(inplace=True),
            nn.Linear(hdim, hdim),
        )
        self.seq_pos_emb = SinusoidalPosEmb(hdim)   # chunk 内序号（加性）

        # ---------------- 主干 ----------------
        # 自注意力是在 [动作 ; context] 拼接后的整条序列上做的，所以叫 joint
        self.joint_self_attn = FFWSelfAttentionLayers(
            hdim, num_heads, num_layers=2, use_adaln=True
        )
        self.cross_attn = FFWCrossAttentionLayers(
            hdim, num_heads, num_layers=1, use_adaln=True
        )
        # 注意力是全局的、置换等变的，这一层补回相邻动作之间的局部时序结构
        self.local_conv = ConditionalResidualBlock1D(
            in_channels=hdim, out_channels=hdim, cond_dim=hdim,
            kernel_size=3, n_groups=num_heads, cond_predict_scale=True,
        )

        # ---------------- 输出头 ----------------
        # 位姿和夹爪分开：夹爪是准二值量，和连续位姿共用一个头会互相拖累
        self.pose_head = nn.Sequential(
            nn.Linear(hdim, hdim), nn.ReLU(inplace=True),
            nn.Linear(hdim, hdim), nn.ReLU(inplace=True),
            nn.Linear(hdim, action_dim - 1),
        )
        self.gripper_head = nn.Sequential(
            nn.Linear(hdim, hdim), nn.ReLU(inplace=True),
            nn.Linear(hdim, hdim), nn.ReLU(inplace=True),
            nn.Linear(hdim, 1),
        )

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        context: Tensor,
        context_pe: Tensor,
        context_mask: Optional[Tensor] = None,
        current_gripper_embed: Optional[Tensor] = None,
        fp16: bool = False,
    ) -> Tensor:
        """
        Args:
            x:          (B, Ta, action_dim)      含噪动作 chunk
            t:          (B,)                  去噪时刻
            context:    (B, Lc, hdim)         StateEncoder 输出
            context_pe: (B, Lc, head_dim, 2)  context 的 RoPE code
            context_mask:          (B, Lc)    None 表示全部有效
            current_gripper_embed: (B, hdim)  当前夹爪状态，并入全局条件

        Returns:
            (B, Ta, action_dim)  预测的 eps / 速度场，布局同输入
        """
        B, Ta, _ = x.shape
        xyz = x[..., :3]                       # 兼作 3D 坐标的那 3 维

        # ---- 全局条件：去噪时刻（+ 夹爪状态）----
        t_emb = self.t_embed(t)                                    # (B, hdim)
        if current_gripper_embed is not None:
            t_emb = t_emb + current_gripper_embed

        # ---- 动作 token：内容投影 + chunk 内序号（加性）----
        h = self.x_proj(x[..., : self.action_dim])                    # (B, Ta, hdim)
        h = h + self.seq_pos_emb(torch.arange(Ta, device=x.device).to(h))[None]

        # ---- 拼成一条序列，动作和 context 一起做自注意力 ----
        h_joint = torch.cat([h, context], dim=1)                   # (B, Ta+Lc, hdim)
        pe_joint = torch.cat([self.rope3d(xyz), context_pe], dim=1)
        if context_mask is not None:
            mask_joint = torch.cat([context_mask.new_ones((B, Ta)), context_mask], dim=1)
        else:
            mask_joint = None

        with torch.autocast(x.device.type, torch.bfloat16 if fp16 else torch.float32):
            h_joint = self.joint_self_attn(
                query=h_joint,
                query_pe=pe_joint,
                query_mask=mask_joint,
                film=t_emb,
            )[-1]

            # 切回动作段；context 段不再单独取用，但仍作为下面 cross-attn 的 value
            h = h_joint[:, :Ta]

            # query 补上绝对位置（加性），value 是整条 [动作 ; context]
            # —— 所以这一步既读 context，也让动作再看一次自己
            h = self.cross_attn(
                query=h + self.xyz_proj(xyz),
                value=h_joint,
                value_mask=mask_joint,
                film=t_emb,
            )[-1]

        # ---- 局部时序：(B,L,C) -> (B,C,L) 过 1D 卷积再转回来 ----
        h = rearrange(h, "b l c -> b c l")
        h = self.local_conv(h, t_emb)
        h = rearrange(h, "b c l -> b l c")

        return torch.cat([self.pose_head(h), self.gripper_head(h)], dim=-1)

    # 旧调用名的别名；注意参数名/顺序已变，关键字调用要跟着改
    action_pred = forward
class ActionDecoderMamba(nn.Module):
    """ActionDecoder 的 Mamba 版：只换序列算子，其余原样保留。

    和 ActionDecoder 的差异只有一处——把 `joint_self_attn`(2 层自注意力) +
    `local_conv`(1 层 FiLM 卷积) 换成一个在 hdim 特征空间上跑的 Mamba U-Net：

        ActionDecoder       x_proj -> [joint_self_attn x2] -> cross_attn -> local_conv -> heads
        ActionDecoderMamba  x_proj -> cross_attn -> [MambaUNet1D] -> heads

    x_proj / seq_pos_emb / xyz_proj / t_embed / cross_attn / pose_head / gripper_head
    全部逐字保留，所以两者的输入投影、条件注入和输出头完全一致，
    唯一变化的自变量就是「全局注意力 vs SSM 扫描」。这是能把结论归因清楚的前提。

    为什么 U-Net 跑在 hdim 而不是 action_dim 上：
        跑在 action_dim 上就等于连输入投影和输出头一起换掉了，届时涨跌无法归因。
        `MambaUNet1D(action_dim=hdim, ...)` 里的 `action_dim` 只是「序列的通道数」，
        传 hdim 就是把它当作 seq2seq 主干用。

    三个已知的设计取舍：
      1. `down_dims` 必须只有 2 级，chunk=50 才不用 padding（50 -> 25 -> 50）。
         3 级要求能被 4 整除，50 不行；而且 2 级的扫描序列更长（50/25 vs 52/26/13），
         SSM 在短序列上本来就没什么可做的。
      2. context 仍由 `cross_attn` 以 token 级注入，所以**不需要**再给 U-Net 补
         cross-attention FiLM。U-Net 自带的那条 mean-pool 全局 FiLM 只是附加信号。
      3. `rope3d` 是乘性 PE，只能活在注意力里，去掉自注意力后这一路会变弱。
         `use_attn_pe=True` 会把它和 `context_pe` 一起挂到仅存的 cross_attn 上，
         补回 ActionDecoder 里由 `joint_self_attn` 承担的那部分位置信息；
         设成 False 则注意力路径上完全没有 PE，只剩加性的 xyz_proj / seq_pos_emb。
    """

    def __init__(
        self,
        hdim: int,
        num_heads: int,
        action_dim: int = 8,
        down_dims: Sequence[int] = (256, 512),
        d_state: int = 16,
        kernel_size: int = 3,
        n_groups: int = 8,
        use_attn_pe: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.use_attn_pe = use_attn_pe
        self.downsample_factor = 2 ** (len(down_dims) - 1)

        # ---------------- 以下与 ActionDecoder 完全一致 ----------------
        self.x_proj = nn.Sequential(
            nn.Linear(action_dim, hdim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hdim, hdim),
        )
        self.xyz_proj = nn.Sequential(
            nn.Linear(3, hdim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hdim, hdim),
        )
        self.rope3d = RoPE(hdim, position_dim=3, num_heads=num_heads)

        self.t_embed = nn.Sequential(
            SinusoidalPosEmb(hdim),
            nn.Linear(hdim, hdim),
            nn.ReLU(inplace=True),
            nn.Linear(hdim, hdim),
        )
        self.seq_pos_emb = SinusoidalPosEmb(hdim)

        self.cross_attn = FFWCrossAttentionLayers(
            hdim, num_heads, num_layers=1, use_adaln=True
        )

        self.pose_head = nn.Sequential(
            nn.Linear(hdim, hdim), nn.ReLU(inplace=True),
            nn.Linear(hdim, hdim), nn.ReLU(inplace=True),
            nn.Linear(hdim, action_dim - 1),
        )
        self.gripper_head = nn.Sequential(
            nn.Linear(hdim, hdim), nn.ReLU(inplace=True),
            nn.Linear(hdim, hdim), nn.ReLU(inplace=True),
            nn.Linear(hdim, 1),
        )
        # ---------------- 唯一的改动 ----------------
        # 顶掉 joint_self_attn + local_conv。in/out 都是 hdim，序列长度 Ta 不变。
        # t_emb 已经是 hdim，所以 time_dim=hdim；cond 是 context，cond_dim=hdim。
        # 想和 ActionDecoder 对齐参数量就调 down_dims，别动 hdim。
        self.backbone = MambaUNet1D(
            action_dim=hdim,
            cond_dim=hdim,
            down_dims=tuple(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            d_state=d_state,
            time_dim=hdim,
        )

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        context: Tensor,
        context_pe: Tensor,
        context_mask: Optional[Tensor] = None,
        current_gripper_embed: Optional[Tensor] = None,
        fp16: bool = False,
    ) -> Tensor:
        """签名与 ActionDecoder.forward 完全相同，可直接对调。

        Args:
            x:          (B, Ta, action_dim)      含噪动作 chunk
            t:          (B,)                  去噪时刻
            context:    (B, Lc, hdim)         StateEncoder 输出
            context_pe: (B, Lc, head_dim, 2)  context 的 RoPE code
            context_mask:          (B, Lc)    None 表示全部有效
            current_gripper_embed: (B, hdim)  当前夹爪状态，并入全局条件

        Returns:
            (B, Ta, action_dim)  预测的 eps / 速度场，布局同输入
        """
        B, Ta, _ = x.shape
        assert Ta % self.downsample_factor == 0, (
            f"chunk 长度 {Ta} 不能被 {self.downsample_factor} 整除，U-Net 上采样后"
            f"长度会对不上；Ta=50 请用 2 级 down_dims"
        )
        xyz = x[..., :3]

        # ---- 全局条件：去噪时刻（+ 夹爪状态）----
        t_emb = self.t_embed(t)
        if current_gripper_embed is not None:
            t_emb = t_emb + current_gripper_embed

        # ---- 动作 token：内容投影 + chunk 内序号（加性）----
        h = self.x_proj(x[..., : self.action_dim])
        h = h + self.seq_pos_emb(torch.arange(Ta, device=x.device).to(h))[None]

        # ---- 交叉注意力注入 context ----
        # value 沿用 ActionDecoder 的构造：[动作 ; context]，动作也能再看一次自己。
        # 区别只是这里的动作段没经过自注意力（那一层已被 U-Net 顶掉）。
        value = torch.cat([h, context], dim=1)
        if context_mask is not None:
            value_mask = torch.cat([context_mask.new_ones((B, Ta)), context_mask], dim=1)
        else:
            value_mask = None

        if self.use_attn_pe:
            query_pe = self.rope3d(xyz)
            value_pe = torch.cat([query_pe, context_pe], dim=1)
        else:
            query_pe = value_pe = None

        with torch.autocast(x.device.type, torch.bfloat16 if fp16 else torch.float32):
            h = self.cross_attn(
                query=h + self.xyz_proj(xyz),
                value=value,
                query_pe=query_pe,
                value_pe=value_pe,
                value_mask=value_mask,
                film=t_emb,
            )[-1]

        # ---- Mamba U-Net 做时序建模 ----
        # 它内部自己转成 channel-first 再转回来，这里不用 rearrange。
        # context 会被 mean-pool 成一个向量走全局 FiLM，作为 t_emb 之外的附加条件；
        # token 级的 context 信息上面 cross_attn 已经注入过了。
        h = self.backbone(h, t_emb, context)

        return torch.cat([self.pose_head(h), self.gripper_head(h)], dim=-1)
