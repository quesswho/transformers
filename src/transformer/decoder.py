import math
import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .config import ModelConfig
from .layers import FeedForward, PositionalEncoding, RMSNorm


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(config.d_model, config.nhead, config.dropout)
        self.cross_attn = MultiHeadAttention(config.d_model, config.nhead, config.dropout)
        self.ff = FeedForward(config.d_model, config.d_ff, config.dropout)
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        self.norm3 = RMSNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # We use Pre-norm as it produces more stable training
        normed = self.norm1(x)
        self_out, present_kv = self.self_attn(normed, normed, normed, tgt_mask, past_kv)
        x = x + self.dropout(self_out)
        cross_out, _ = self.cross_attn(self.norm2(x), enc_output, enc_output, src_mask)
        x = x + self.dropout(cross_out)
        x = x + self.dropout(self.ff(self.norm3(x)))
        return x, present_kv


class Decoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.dropout, config.max_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.d_model)
        self.d_model = config.d_model

    def forward(
        self,
        tgt: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        offset = past_key_values[0][0].size(2) if past_key_values is not None else 0
        x = self.pos_encoding(self.embedding(tgt) * math.sqrt(self.d_model), offset=offset)
        present_key_values = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, kv = layer(x, enc_output, src_mask, tgt_mask, past_kv)
            present_key_values.append(kv)
        return self.norm(x), present_key_values
