import torch
import torch.nn as nn
from .config import EncoderDecoderConfig, ModelConfig
from .encoder import Encoder
from .decoder import Decoder


def make_src_mask(src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    tgt_len = tgt.size(1)
    causal = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
    padding = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    return causal & padding


def _init_weights(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        if p.dim() <= 1:
            continue
        if name.endswith("w_qkv.weight"):
            # Xavier each projection at its own (d_model, d_model) fan, not the
            # fused (3*d_model, d_model) shape, which would shrink the scale.
            for chunk in p.data.chunk(3, dim=0):
                nn.init.xavier_uniform_(chunk)
        else:
            nn.init.xavier_uniform_(p)


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, config: EncoderDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.pad_idx = config.pad_idx
        self.encoder = Encoder(config.encoder_config())
        self.decoder = Decoder(config.decoder_config())
        self.projection = nn.Linear(config.d_model, config.tgt_vocab_size)
        _init_weights(self)

    def count_parameters(self) -> dict[str, int]:
        def n(m): return sum(p.numel() for p in m.parameters())
        enc_norms = n(self.encoder.norm) + sum(
            n(l.norm1) + n(l.norm2) for l in self.encoder.layers
        )
        dec_norms = n(self.decoder.norm) + sum(
            n(l.norm1) + n(l.norm2) + n(l.norm3) for l in self.decoder.layers
        )
        return {
            "embeddings":         n(self.encoder.embedding) + n(self.decoder.embedding),
            "encoder_attention":  sum(n(l.self_attn) for l in self.encoder.layers),
            "encoder_ffn":        sum(n(l.ff) for l in self.encoder.layers),
            "encoder_norms":      enc_norms,
            "decoder_self_attn":  sum(n(l.self_attn) for l in self.decoder.layers),
            "decoder_cross_attn": sum(n(l.cross_attn) for l in self.decoder.layers),
            "decoder_ffn":        sum(n(l.ff) for l in self.decoder.layers),
            "decoder_norms":      dec_norms,
            "projection":         n(self.projection),
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
        self.max_len = config.max_len
        self.encoder = Encoder(config)
        self.projection = nn.Linear(config.d_model, config.vocab_size)
        _init_weights(self)

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
        norms = n(self.encoder.norm) + sum(
            n(l.norm1) + n(l.norm2) for l in self.encoder.layers
        )
        return {
            "embeddings": n(self.encoder.embedding),
            "attention":  sum(n(l.self_attn) for l in self.encoder.layers),
            "ffn":        sum(n(l.ff) for l in self.encoder.layers),
            "norms":      norms,
            "projection": n(self.projection),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.encoder(x, is_causal=True)
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
                ctx = idx[:, -self.max_len:]
                hidden, past_key_values = self.encoder(ctx, is_causal=True)
            else:
                T = past_key_values[0][0].size(2)
                hidden, past_key_values = self.encoder(idx[:, -1:], src_mask=None, past_key_values=past_key_values, offset=T)
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
