import math
import torch
from typing import List
from torch import nn, Tensor
from models.layers.mha import MySimpleMHA


class LoraLinear(nn.Module):
    def __init__(self, lin: nn.Linear, r: int):
        super().__init__()
        self.r = r
        self.lin = lin

        for p in self.lin.parameters():
            p.requires_grad_(False)

        self.A = nn.Parameter(torch.zeros(lin.in_features, r).to(lin.weight.device))
        self.B = nn.Parameter(torch.zeros(r, lin.out_features).to(lin.weight.device))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A, math.sqrt(5))
        with torch.no_grad():
            scaling = 1.0 / self.r
            self.A.mul_(scaling)

    def forward(self, x: Tensor):
        return self.lin(x) + (x @ self.A) @ self.B


def replace_with_lora_linear(model: nn.Module, r: int):
    num_replaced = 0
    for name, m in model.named_modules():
        if isinstance(m, MySimpleMHA):
            for key in ["to_q", "to_k", "to_v", "to_qk", "to_kv", "to_qkv"]:
                if hasattr(m, key):
                    num_replaced += 1
                    print("[INFO] [{}] Replace {}".format(num_replaced, name))
                    n = len(key.replace("to_", ""))
                    lora_linear = LoraLinear(getattr(m, key), r * n)
                    setattr(m, key, lora_linear)
    return model, num_replaced


# Kept trainable alongside the LoRA factors. LayerNorm affine params and biases are the
# standard companions to LoRA ("LoRA + bias/norm tuning"): they cost almost nothing and
# absorb the distribution shift that low-rank updates alone adapt to slowly. The two
# learned embeddings are tiny (64x768 and 768) and are the model's only handle on
# "which query slot" and "which camera is the main one".
LORA_KEEP_TRAINABLE = ("norm", ".bias", "qformer.queries", "main_cam_embed")


def setup_lora(module: nn.Module, rank: int, keep_trainable=LORA_KEEP_TRAINABLE):
    """Inject LoRA into `module`'s attention projections and freeze everything else.

    MUST be called AFTER the pretrained weights are loaded: `replace_with_lora_linear`
    wraps each `nn.Linear` in a `LoraLinear`, which renames `...to_q.weight` to
    `...to_q.lin.weight` in the state_dict. Loading a plain (non-LoRA) checkpoint into
    an already-injected model would therefore fail on every attention projection.

    Injection is a no-op at the function level: `LoraLinear.B` is zero-initialised, so
    `lin(x) + (x @ A) @ B` is bit-identical to `lin(x)` until training moves B.

    End to end the context output can still drift by ~1e-6 in float32. That is not a
    logic error: `lin(x) + 0` allocates a fresh tensor, which lands at a different
    address and can take a different blocking path through the matmul/SDPA kernels.
    Verified: the drift shrinks when the thread count is pinned to 1, parameters stay
    bit-identical, and every LoraLinear output matches its base Linear exactly. For
    scale, training runs in bfloat16, whose epsilon is ~4000x larger than this drift.

    Args:
        module: subtree to adapt (here: `actor.context_encoder`)
        rank: LoRA rank r. The effective rank of a fused projection is `r * n`, where n
            is the number of matrices it fuses (to_qkv -> 3r), so each individual
            projection gets rank r.
        keep_trainable: name substrings that stay trainable despite the freeze

    Returns:
        (num_replaced, num_trainable_params)
    """
    if rank <= 0:
        raise ValueError("lora rank must be positive, got {}".format(rank))

    module, num_replaced = replace_with_lora_linear(module, rank)

    # replace_with_lora_linear only freezes the Linears it wrapped; freeze the rest here
    # (FFNs, projections, the QFormer's non-attention weights, ...).
    lora_params = {id(p) for p in get_lora_parameters(module)}
    num_trainable = 0
    for name, p in module.named_parameters():
        if id(p) in lora_params or any(pat in name for pat in keep_trainable):
            p.requires_grad_(True)
            num_trainable += p.numel()
        else:
            p.requires_grad_(False)

    total = sum(p.numel() for p in module.parameters())
    print("[INFO] LoRA rank={}: wrapped {} attention projections; {:.2f}M / {:.2f}M "
          "params trainable in this subtree ({:.1f}%)"
          .format(rank, num_replaced, num_trainable / 1e6, total / 1e6,
                  num_trainable / total * 100))
    return num_replaced, num_trainable


def get_lora_parameters(model: nn.Module):
    params: List[nn.Parameter] = []
    for m in model.modules():
        if isinstance(m, LoraLinear):
            params.append(m.A)
            params.append(m.B)
    return params


@torch.no_grad()
def merge_lora_linear(model: nn.Module, inplace: bool = True):

    index = 0

    def _merge_lora_linear_inplace(model: nn.Module, target_module_name=""):
        nonlocal index
        for name, m in model.named_children():
            if isinstance(m, LoraLinear):
                index += 1
                linear = m.lin
                lora_weight = torch.einsum("i r, r o -> o i", m.A, m.B)

                if inplace:
                    linear.weight.copy_(linear.weight + lora_weight)
                    setattr(model, name, linear)
                else:
                    new_linear = nn.Linear(
                        in_features=linear.in_features,
                        out_features=linear.out_features,
                        bias=linear.bias is not None,
                        device=linear.weight.device,
                        dtype=linear.weight.dtype,
                    )
                    new_linear.weight.copy_(linear.weight + lora_weight)
                    if new_linear.bias is not None:
                        new_linear.bias.copy_(linear.bias)
                    setattr(model, name, new_linear)

                print("[INFO] [{}] Merge {}".format(index, target_module_name + "." + name))
            else:
                _merge_lora_linear_inplace(m, target_module_name + "." + name if target_module_name else name)
        return model
    
    merged = _merge_lora_linear_inplace(model, "")
    for m in merged.modules():
        assert not isinstance(m, LoraLinear)
    
    return merged


