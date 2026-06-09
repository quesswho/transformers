import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_k = d_model // nhead
        self.nhead = nhead
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B = query.size(0)

        def project_and_split(linear, x):
            return linear(x).view(B, -1, self.nhead, self.d_k).transpose(1, 2)

        q = project_and_split(self.w_q, query)
        k = project_and_split(self.w_k, key)
        v = project_and_split(self.w_v, value)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        dropout_p = self.dropout.p if self.training else 0.0
        # is_causal lets SDPA pick the flash-attention kernel; an explicit
        # attn_mask forces a slower backend (and SDPA forbids passing both).
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None if is_causal else mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, self.nhead * self.d_k)
        return self.w_o(attn_out), (k, v)
