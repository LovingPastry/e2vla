from torch import nn, Tensor
from typing import Optional, List
from .layers.attn_dn import SelfAttentionLayer, CrossAttentionLayer, FFN, init_xncoder


class DiTBlock(nn.Module):
    def __init__(
        self, 
        hdim: int, 
        num_heads: int, 
        use_adaln: bool = False, 
        pe_type: str = "rope"
    ):
        super().__init__()
        self.self_attn = SelfAttentionLayer(
            hdim, num_heads, 
            use_adaln=use_adaln, 
            bias=True, 
            qk_norm=True, 
            pe_type=pe_type
        )
        self.cross_attn = CrossAttentionLayer(
            hdim, 
            num_heads, 
            use_adaln=False, 
            bias=True, 
            qk_norm=True, 
            pe_type=pe_type
        )
        self.ffn = FFN(
            hdim, 
            hidden_dim=2*hdim, 
            use_adaln=use_adaln
        )
    
    def forward(
        self, 
        x: Tensor,
        x_pe: Optional[Tensor],
        x_mask: Optional[Tensor],
        c: Tensor,
        c_mask: Optional[Tensor],
        film: Optional[Tensor],
        x_pe_inv: Optional[Tensor] = None,
    ):
        # self attn
        x, x_mask = self.self_attn(
            query=x,
            query_pe=x_pe,
            query_mask=x_mask,
            film=film,
            query_pe_inv=x_pe_inv
        )
        # cross attn
        x, c_mask = self.cross_attn(
            query=x,
            value=c,
            value_mask=c_mask
        )
        # ffn
        x = self.ffn(x, film=film)
        return x, x_mask, c_mask


class DiT(nn.Module):
    def __init__(
        self, 
        hdim: int, 
        num_heads: int, 
        num_layers: int, 
        use_adaln: bool = False, 
        pe_type: str = "rope"
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([DiTBlock(hdim, num_heads, use_adaln, pe_type) 
                                     for _ in range(num_layers)])
        self.reset_parameters()
    
    def reset_parameters(self):
        init_xncoder(self.num_layers*2, self.layers)

    def forward(
        self,
        x: Tensor,
        x_pe: Optional[Tensor],
        x_mask: Optional[Tensor],
        conds: List[Tensor],
        cond_masks: Optional[List[Optional[Tensor]]],
        films: Optional[List[Optional[Tensor]]],
        x_pe_inv: Optional[Tensor] = None,
    ):
        """
        - x_pe_inv: inverse of `x_pe`, only used if pe_type == "prope". Computing it
          once outside is much cheaper than inverting the same matrices in every layer.
        """
        NoneType = type(None)

        # wrap to iterable
        if isinstance(conds, Tensor):
            conds = [conds]
        if isinstance(cond_masks, (NoneType, Tensor)):
            cond_masks = [cond_masks]
        if isinstance(films, (NoneType, Tensor)):
            films = [films]

        n_cond_type = len(conds)
        assert n_cond_type <= self.num_layers, (
            "n_cond_type = {}, num_layers = {}"
            .format(n_cond_type, self.num_layers)
        )

        cond_masks = cond_masks.copy()  # shallow copy
        for i in range(self.num_layers):
            x, x_mask, cond_masks[i%n_cond_type] = self.layers[i](
                x=x,
                x_pe=x_pe,
                x_mask=x_mask,
                c=conds[i%n_cond_type],
                c_mask=cond_masks[i%n_cond_type],
                film=films[i%n_cond_type],
                x_pe_inv=x_pe_inv
            )
        return x
