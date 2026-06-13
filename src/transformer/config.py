from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    """Hyperparameters for a single transformer stack (used directly by
    DecoderOnlyTransformer and per-stack inside EncoderDecoderTransformer)."""

    vocab_size: int
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 5000  # context-window cap for generation (not a hard architectural limit under RoPE)
    rope_theta: float = 10000.0  # rotary embedding base frequency
    tie_embeddings: bool = True  # share input embedding and output projection weights

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**d)


@dataclass
class EncoderDecoderConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 5000  # context-window cap for generation (not a hard architectural limit under RoPE)
    rope_theta: float = 10000.0  # rotary embedding base frequency
    pad_idx: int = 0
    tie_embeddings: bool = False  # tie decoder input embedding and output projection

    def encoder_config(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=self.src_vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            max_seq_len=self.max_seq_len,
            rope_theta=self.rope_theta,
        )

    def decoder_config(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=self.tgt_vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            max_seq_len=self.max_seq_len,
            rope_theta=self.rope_theta,
            tie_embeddings=self.tie_embeddings,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EncoderDecoderConfig":
        return cls(**d)
