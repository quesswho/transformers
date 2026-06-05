import math
import torch
import torch.nn as nn


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return torch.matmul(weights, value), weights


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
    ) -> torch.Tensor:
        B = query.size(0)

        def project_and_split(linear, x):
            return linear(x).view(B, -1, self.nhead, self.d_k).transpose(1, 2)

        q = project_and_split(self.w_q, query)
        k = project_and_split(self.w_k, key)
        v = project_and_split(self.w_v, value)

        attn_out, _ = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, self.nhead * self.d_k)
        return self.w_o(attn_out)
