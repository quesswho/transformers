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
    max_len: int = 5000

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
    max_len: int = 5000
    pad_idx: int = 0

    def encoder_config(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=self.src_vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            max_len=self.max_len,
        )

    def decoder_config(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=self.tgt_vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            max_len=self.max_len,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EncoderDecoderConfig":
        return cls(**d)
