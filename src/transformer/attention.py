import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RotaryEmbedding


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dropout: float = 0.1,
        rope: RotaryEmbedding | None = None,
    ) -> None:
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_k = d_model // nhead
        self.nhead = nhead
        self.d_model = d_model
        # Q, K, V projections fused into one matmul: at small d_model the three
        # separate GEMMs are launch-bound, so one 3x-wide GEMM is ~free.
        self.w_qkv = nn.Linear(d_model, 3 * d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # Rotary position embedding, shared across layers. None for cross-attention,
        # which spans two coordinate frames and so carries no rotary position.
        self.rope = rope

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        is_causal: bool = False,
        offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B = query.size(0)
        d = self.d_model
        w, b = self.w_qkv.weight, self.w_qkv.bias

        if query is key and key is value:
            q, k, v = self.w_qkv(query).chunk(3, dim=-1)
        elif key is value:
            # Cross-attention: query comes from a different sequence, but K and
            # V still share an input so their projections stay fused.
            q = F.linear(query, w[:d], b[:d])
            k, v = F.linear(key, w[d:], b[d:]).chunk(2, dim=-1)
        else:
            q = F.linear(query, w[:d], b[:d])
            k = F.linear(key, w[d : 2 * d], b[d : 2 * d])
            v = F.linear(value, w[2 * d :], b[2 * d :])

        def split_heads(x):
            return x.view(B, -1, self.nhead, self.d_k).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Rotate the new q and k by their absolute positions before they hit the
        # cache, so the stored keys are already rotated and concatenation Just Works.
        if self.rope is not None:
            q, k = self.rope(q, k, offset)

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
