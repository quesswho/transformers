import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused kernel: one launch instead of ~5 (pow/mean/add/sqrt/div/mul),
        # and it computes the statistics in fp32 even under bf16 autocast.
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


class FeedForward(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_value = nn.Linear(d_model, d_ff, bias=False)
        self.w_out = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(self.dropout(self.act(self.w_gate(x)) * self.w_value(x)))


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (RoPE). Rotates query/key vectors by an angle
    proportional to their absolute position, which makes the dot product depend
    only on the *relative* offset between positions. Applied inside attention on
    the per-head vectors, so it leaves the SDPA flash path intact.

    A single instance is shared across all attention layers in a stack: the only
    state is the `inv_freq` buffer, and cos/sin are recomputed per call (cheap
    next to the attention matmuls) so arbitrary KV-cache offsets just work."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        assert dim % 2 == 0, "rotary dim (d_model // nhead) must be even"
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k: (B, nhead, T, d_k). In self-attention the new q and k share the
        # same length and absolute positions offset .. offset + T.
        T = q.size(-2)
        pos = torch.arange(offset, offset + T, device=q.device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq)          # (T, d_k/2)
        emb = torch.cat((freqs, freqs), dim=-1)           # (T, d_k)
        cos = emb.cos()[None, None, :, :]                 # (1, 1, T, d_k)
        sin = emb.sin()[None, None, :, :]
        # Rotation runs in fp32 (cos/sin promote q,k) then casts back, so it is
        # numerically stable under bf16/fp16 autocast.
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot.type_as(q), k_rot.type_as(k)
