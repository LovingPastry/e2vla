import torch
from typing import Optional
from torch import nn, Tensor
from .layers.utils import concat_mask
from .layers.attn_dn import SelfAttentionLayer, CrossAttentionLayer, FFN, init_xncoder


class QBlockITM(nn.Module):
    def __init__(self, hdim: int, num_heads: int):
        super().__init__()
        self.self_attn_text = SelfAttentionLayer(hdim, num_heads, bias=True, qk_norm=True)
        self.cross_attn_vision = CrossAttentionLayer(hdim, num_heads, bias=True, qk_norm=True)
        self.ffn_query = FFN(hdim, 2*hdim)
        self.ffn_text = FFN(hdim, 2*hdim)
    
    def forward(
        self, 
        query: Tensor, 
        x_vision: Tensor, 
        mask_vision: Optional[Tensor], 
        x_text: Tensor,
        mask_qt: Optional[Tensor],
    ):
        """
        Args:
            query (Tensor): (B, Lq, C)
            x_vision (Tensor): (B, Lv, C)
            mask_vision (Optional[Tensor]): (B, Lv) or None
            x_text (Tensor): (B, Lt, C)
            mask_qt (Optional[Tensor]): (B, Lq+Lt) or None
        
        Returns
        -------
            query (Tensor): (B, Lq, C)
            x_text (Tensor): (B, Lt, C)
            mask_vision (Optional[Tensor]): (B, Lv) or None
            mask_qt (Optional[Tensor]): (B, Lq+Lt) or None
        """
        qt = torch.cat([query, x_text], dim=1)
        qt, mask_qt = self.self_attn_text(
            query=qt, 
            query_mask=mask_qt
        )
        
        query, x_text = torch.split(qt, [query.shape[1], x_text.shape[1]], dim=1)
        x_text = self.ffn_text(x_text)

        query, mask_vision = self.cross_attn_vision(
            query=query, 
            value=x_vision, 
            value_mask=mask_vision
        )
        query = self.ffn_query(query)
        return query, x_text, mask_vision, mask_qt


class QBlockVision(nn.Module):
    """`QBlockITM` with the text branch removed -- the vision-action (VA) counterpart.

    Two things go away with language, and only one of them is a parameter:

    * `self_attn_text` ran over the concatenation [query ; text], so it mixed the two
      modalities. Without text it is a plain self-attention over the queries, which is
      what `self_attn` is here -- the same module, the same shapes, the same cost per
      query token. No parameter is lost.
    * `ffn_text` fed the text tokens onward to the next block. There are no text tokens
      to feed, so it is not built. That is the whole parameter difference between this
      block and `QBlockITM` (~2.4M at hdim=768), and it is why the "sa" context encoder
      lands within a few percent of the vision-language one rather than exactly on it.

    The cross-attention into the vision tokens -- the part that actually does the
    compression from Ncam*Lv patches down to `num_queries` -- is untouched.
    """

    def __init__(self, hdim: int, num_heads: int):
        super().__init__()
        self.self_attn = SelfAttentionLayer(hdim, num_heads, bias=True, qk_norm=True)
        self.cross_attn_vision = CrossAttentionLayer(hdim, num_heads, bias=True, qk_norm=True)
        self.ffn_query = FFN(hdim, 2*hdim)

    def forward(
        self,
        query: Tensor,
        x_vision: Tensor,
        mask_vision: Optional[Tensor],
    ):
        """
        Args:
            query (Tensor): (B, Lq, C)
            x_vision (Tensor): (B, Lv, C)
            mask_vision (Optional[Tensor]): (B, Lv) or None

        Returns
        -------
            query (Tensor): (B, Lq, C)
            mask_vision (Optional[Tensor]): (B, Lv) or None
        """
        query, _ = self.self_attn(query=query, query_mask=None)
        query, mask_vision = self.cross_attn_vision(
            query=query,
            value=x_vision,
            value_mask=mask_vision
        )
        query = self.ffn_query(query)
        return query, mask_vision


class QFormerVision(nn.Module):
    """Language-free QFormer: Ncam*Lv patch tokens -> `num_queries` context tokens.

    Same role and same output shape as `QFormerITM`'s `query`, so the diffusion head
    downstream cannot tell the two apart. See `QBlockVision` for what the missing text
    branch costs.
    """

    def __init__(self, hdim: int, num_heads: int, num_layers: int, num_queries: int):
        super().__init__()
        self.num_layers = num_layers
        # Deliberately the same attribute name as QFormerITM's: `train_utils/lora.py`
        # keeps "qformer.queries" trainable under LoRA by substring match.
        self.queries = nn.Parameter(torch.randn(1, num_queries, hdim))
        self.layers = nn.ModuleList([QBlockVision(hdim, num_heads)
                                     for _ in range(num_layers)])
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.queries, std=0.02)
        init_xncoder(self.num_layers*2, self.layers)

    def forward(
        self,
        x_vision: Tensor,
        mask_vision: Optional[Tensor],
    ):
        """
        Args:
            x_vision (Tensor): (B, Lv, C)
            mask_vision (Optional[Tensor]): (B, Lv) or None

        Returns:
            query (Tensor): (B, Lq, C)
        """
        B = x_vision.shape[0]
        query = self.queries.expand(B, -1, -1)

        for layer in self.layers:
            query, mask_vision = layer(
                query=query,
                x_vision=x_vision,
                mask_vision=mask_vision
            )

        return query


class QFormerITM(nn.Module):
    def __init__(self, hdim: int, num_heads: int, num_layers: int, num_queries: int):
        super().__init__()
        self.num_layers = num_layers
        self.queries = nn.Parameter(torch.randn(1, num_queries, hdim))
        self.layers = nn.ModuleList([QBlockITM(hdim, num_heads) 
                                     for _ in range(num_layers)])
        self.reset_parameters()
    
    def reset_parameters(self):
        ### init params
        nn.init.trunc_normal_(self.queries, std=0.02)
        init_xncoder(self.num_layers*2, self.layers)

    def forward(
        self, 
        x_vision: Tensor, 
        mask_vision: Optional[Tensor], 
        x_text: Tensor,
        mask_text: Optional[Tensor]
    ):
        """
        Args:
            x_vision (Tensor): (B, Lv, C)
            mask_vision (Optional[Tensor]): (B, Lv) or None
            x_text (Tensor): (B, Lt, C)
            mask_text (Optional[Tensor]): (B, Lt) or None
        
        Returns:
            query (Tensor): (B, Lq, C)
            x_text (Tensor): (B, Lt, C)
            mask_qt (Optional[Tensor]): (B, Lq+Lt) or None
        """
        B = x_vision.shape[0]
        query = self.queries.expand(B, -1, -1)
        mask_qt = concat_mask(mask0=None, mask1=mask_text,
                              L0=query.shape[1], L1=x_text.shape[1])
        
        for layer in self.layers:
            query, x_text, mask_vision, mask_qt = layer(
                query=query,
                x_vision=x_vision,
                mask_vision=mask_vision,
                x_text=x_text,
                mask_qt=mask_qt
            )
        
        return query, x_text, mask_qt
