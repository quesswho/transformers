import math

import torch
import torch.nn as nn
from .config import EncoderDecoderConfig, ModelConfig
from .stack import TransformerStack
from .decoder import Decoder


def make_src_mask(src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    tgt_len = tgt.size(1)
    causal = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
    padding = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    return causal & padding


def _init_weights(model: nn.Module, num_layers: int) -> None:
    """GPT-2-style initialization.

    Weight matrices (and embeddings) are drawn from N(0, 0.02); biases are
    zeroed and 1-D parameters (RMSNorm gains) are left untouched. The residual
    output projections (attention ``w_o``, feed-forward ``w_out``) are then
    scaled by 1/sqrt(2 * num_layers): each block adds two residual contributions
    to the stream, so without this down-scaling the residual-stream variance
    grows with depth. Per-element normal init is shape-independent, so the fused
    ``w_qkv`` needs no special-casing.
    """
    for name, p in model.named_parameters():
        if name.endswith("bias"):
            nn.init.zeros_(p)
        elif p.dim() >= 2:
            nn.init.normal_(p, mean=0.0, std=0.02)

    scale = 1.0 / math.sqrt(2 * num_layers)
    for name, p in model.named_parameters():
        if name.endswith(("w_o.weight", "w_out.weight")):
            p.data.mul_(scale)


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, config: EncoderDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.pad_idx = config.pad_idx
        self.encoder = TransformerStack(config.encoder_config())
        self.decoder = Decoder(config.decoder_config())
        self.projection = nn.Linear(config.d_model, config.tgt_vocab_size)
        _init_weights(self, config.num_layers)
        if config.tie_embeddings:
            self.projection.weight = self.decoder.embedding.weight

    def count_parameters(self) -> dict[str, int]:
        def n(m): return sum(p.numel() for p in m.parameters())
        enc_norms = n(self.encoder.norm) + sum(
            n(l.norm1) + n(l.norm2) for l in self.encoder.layers
        )
        dec_norms = n(self.decoder.norm) + sum(
            n(l.norm1) + n(l.norm2) + n(l.norm3) for l in self.decoder.layers
        )
        projection = n(self.projection)
        if self.projection.weight is self.decoder.embedding.weight:
            # Tied: the weight is shared with the embedding, count it once there.
            projection -= self.projection.weight.numel()
        return {
            "embeddings":         n(self.encoder.embedding) + n(self.decoder.embedding),
            "encoder_attention":  sum(n(l.self_attn) for l in self.encoder.layers),
            "encoder_ffn":        sum(n(l.ff) for l in self.encoder.layers),
            "encoder_norms":      enc_norms,
            "decoder_self_attn":  sum(n(l.self_attn) for l in self.decoder.layers),
            "decoder_cross_attn": sum(n(l.cross_attn) for l in self.decoder.layers),
            "decoder_ffn":        sum(n(l.ff) for l in self.decoder.layers),
            "decoder_norms":      dec_norms,
            "projection":         projection,
        }

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        enc_output, _ = self.encoder(src, src_mask)
        return enc_output

    def decode(
        self,
        tgt: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor | None,
        past_key_values: list | None = None,
    ) -> tuple[torch.Tensor, list]:
        return self.decoder(tgt, enc_output, src_mask, tgt_mask, past_key_values)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_mask = make_src_mask(src, self.pad_idx)
        tgt_mask = make_tgt_mask(tgt, self.pad_idx)
        enc_output = self.encode(src, src_mask)
        dec_output, _ = self.decode(tgt, enc_output, src_mask, tgt_mask)
        return self.projection(dec_output)

    @torch.inference_mode()
    def generate(
        self,
        src: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int = 50,
    ) -> list[int]:
        self.eval()
        src_mask = make_src_mask(src, self.pad_idx)
        enc_output = self.encode(src, src_mask)
        tgt = torch.tensor([[sos_idx]], device=src.device)
        past_key_values = None
        result = []
        for _ in range(max_len):
            tgt_input = tgt if past_key_values is None else tgt[:, -1:]
            tgt_mask = make_tgt_mask(tgt, self.pad_idx) if past_key_values is None else None
            dec_output, past_key_values = self.decode(tgt_input, enc_output, src_mask, tgt_mask, past_key_values)
            next_token = self.projection(dec_output[:, -1, :]).argmax(dim=-1).item()
            if next_token == eos_idx:
                break
            result.append(next_token)
            tgt = torch.cat([tgt, torch.tensor([[next_token]], device=src.device)], dim=1)
        return result


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.stack = TransformerStack(config)
        self.projection = nn.Linear(config.d_model, config.vocab_size)
        _init_weights(self, config.num_layers)
        if config.tie_embeddings:
            self.projection.weight = self.stack.embedding.weight

    @classmethod
    def from_checkpoint(
        cls, path: str, map_location=None
    ) -> tuple["DecoderOnlyTransformer", dict]:
        """Rebuild a model from a checkpoint saved by the training scripts.

        Returns the model and the raw checkpoint dict so callers can restore
        the tokenizer, optimizer state, and step counter.
        """
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(ModelConfig.from_dict(checkpoint["config"]))
        model.load_state_dict(checkpoint["model_state_dict"])
        return model, checkpoint

    def count_parameters(self) -> dict[str, int]:
        def n(m): return sum(p.numel() for p in m.parameters())
        norms = n(self.stack.norm) + sum(
            n(l.norm1) + n(l.norm2) for l in self.stack.layers
        )
        projection = n(self.projection)
        if self.projection.weight is self.stack.embedding.weight:
            # Tied: the weight is shared with the embedding, count it once there.
            projection -= self.projection.weight.numel()
        return {
            "embeddings": n(self.stack.embedding),
            "attention":  sum(n(l.self_attn) for l in self.stack.layers),
            "ffn":        sum(n(l.ff) for l in self.stack.layers),
            "norms":      norms,
            "projection": projection,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.stack(x, is_causal=True)
        return self.projection(hidden)

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        eos_idx: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        past_key_values = None
        for _ in range(max_new_tokens):
            if past_key_values is None:
                ctx = idx[:, -self.max_seq_len:]
                hidden, past_key_values = self.stack(ctx, is_causal=True)
            else:
                T = past_key_values[0][0].size(2)
                hidden, past_key_values = self.stack(idx[:, -1:], past_key_values=past_key_values, offset=T)
            logits = self.projection(hidden)[:, -1, :]
            if temperature == 1.0:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
            if eos_idx is not None and (next_tok == eos_idx).all():
                break
        return idx


class GPTBERT(DecoderOnlyTransformer):
    """A decoder-only transformer trained jointly as a GPT and a BERT
    (Charpentier & Samuel, 2024).

    The architecture is identical to ``DecoderOnlyTransformer`` -- the same
    ``stack`` + tied ``projection`` -- so its checkpoints are interchangeable
    with the causal model for generation, HF export, and zero-shot evaluation.
    The only addition is an ``is_causal`` switch on ``forward``:

    * ``is_causal=True``  -> autoregressive next-token prediction (the GPT path).
    * ``is_causal=False`` -> bidirectional masked-next-token prediction (MNTP,
      the BERT path), where the input has been mask-corrupted by
      ``training.get_mntp_batch``.

    MNTP shifts the masked-LM labels by one so the prediction for a masked token
    is read from the *previous* position's hidden state -- the same output offset
    the causal objective uses -- which is what lets a single output head serve
    both objectives. ``generate`` is inherited unchanged and always runs causally.
    """

    def forward(self, x: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        hidden, _ = self.stack(x, is_causal=is_causal)
        return self.projection(hidden)
