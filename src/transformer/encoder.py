import math
import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .layers import FeedForward, PositionalEncoding


class EncoderLayer(nn.Module):
    def __init__(
        self, d_model: int = 512, nhead: int = 8, d_ff: int = 2048, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def count_parameters(self) -> dict[str, int]:
        def n(m): return sum(p.numel() for p in m.parameters())
        return {
            "embeddings":  n(self.embedding),
            "attention":   sum(n(l.self_attn) for l in self.layers),
            "ffn":         sum(n(l.ff) for l in self.layers),
            "layer_norms": n(self.norm) + sum(n(l.norm1) + n(l.norm2) for l in self.layers),
        }

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.pos_encoding(self.embedding(src) * math.sqrt(self.d_model))
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
