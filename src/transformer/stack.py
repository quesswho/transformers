import math
import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .config import ModelConfig
from .layers import FeedForward, RMSNorm, RotaryEmbedding


class TransformerBlock(nn.Module):
    """Pre-norm self-attention + feed-forward block. Bidirectional or causal
    depending on the is_causal flag, so it serves both the encoder and
    decoder-only paths."""

    def __init__(self, config: ModelConfig, rope: RotaryEmbedding) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(config.d_model, config.nhead, config.dropout, rope)
        self.ff = FeedForward(config.d_model, config.d_ff, config.dropout)
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        is_causal: bool = False,
        offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        normed = self.norm1(x)
        attn_out, present_kv = self.self_attn(normed, normed, normed, mask, past_kv, is_causal, offset)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, present_kv


class TransformerStack(nn.Module):
    """Embedding + N TransformerBlocks + final norm. With is_causal=False this
    is the classic encoder; with is_causal=True it is a GPT-style decoder-only
    stack."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(config.d_model // config.nhead, config.rope_theta)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, self.rope) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.d_model)
        self.d_model = config.d_model

    def count_parameters(self) -> dict[str, int]:
        def n(m): return sum(p.numel() for p in m.parameters())
        return {
            "embeddings":  n(self.embedding),
            "attention":   sum(n(l.self_attn) for l in self.layers),
            "ffn":         sum(n(l.ff) for l in self.layers),
            "layer_norms": n(self.norm) + sum(n(l.norm1) + n(l.norm2) for l in self.layers),
        }

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        offset: int = 0,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        x = self.dropout(self.embedding(tokens) * math.sqrt(self.d_model))
        present_key_values = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, kv = layer(x, mask, past_kv, is_causal, offset)
            present_key_values.append(kv)
        return self.norm(x), present_key_values
