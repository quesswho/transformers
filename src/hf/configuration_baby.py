"""HuggingFace `PretrainedConfig` for the decoder-only transformer.

This is a thin 1:1 mapping of the repo's `ModelConfig` onto the HF config
interface so the model can be loaded with `AutoModelForCausalLM.from_pretrained`
and scored by the BabyLM evaluation harness (lm-eval-harness).
"""

from transformers import PretrainedConfig

from transformer import ModelConfig


class BabyTransformerConfig(PretrainedConfig):
    """HF config carrying the same fields as `ModelConfig`.

    `tie_embeddings` is mirrored onto HF's own `tie_word_embeddings` so the
    base `PreTrainedModel.tie_weights()` machinery stays consistent with the
    inner model's tying.
    """

    model_type = "baby_transformer"

    def __init__(
        self,
        vocab_size: int = 2000,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        rope_theta: float = 10000.0,
        tie_embeddings: bool = True,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.tie_embeddings = tie_embeddings
        # Alias the context window under the name lm-eval-harness probes when it
        # auto-detects a model's max sequence length.
        self.max_position_embeddings = max_seq_len
        # Keep HF's weight-tying flag in lock-step with ours.
        kwargs.setdefault("tie_word_embeddings", tie_embeddings)
        super().__init__(**kwargs)

    def to_model_config(self) -> ModelConfig:
        """Build the repo's native `ModelConfig` from this HF config."""
        return ModelConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            max_seq_len=self.max_seq_len,
            rope_theta=self.rope_theta,
            tie_embeddings=self.tie_embeddings,
        )

    @classmethod
    def from_model_config(cls, config: ModelConfig, **kwargs) -> "BabyTransformerConfig":
        """Build an HF config from the repo's native `ModelConfig`."""
        return cls(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
            tie_embeddings=config.tie_embeddings,
            **kwargs,
        )


# Make `save_pretrained` bundle this file and write an `auto_map` into config.json
# so the checkpoint loads in a fresh process (the eval harness) via
# `AutoConfig.from_pretrained(path, trust_remote_code=True)` — without it the
# in-process `register_auto()` call is invisible to the harness.
BabyTransformerConfig.register_for_auto_class()
