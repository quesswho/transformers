"""HuggingFace integration for the decoder-only transformer.

Exposes `BabyTransformerConfig` / `BabyTransformerForCausalLM` and registers
them with the `Auto*` factories so `AutoConfig` / `AutoModelForCausalLM` resolve
the `"baby_transformer"` model_type locally (without trust_remote_code).
"""

from transformers import AutoConfig, AutoModelForCausalLM

from .configuration_baby import BabyTransformerConfig
from .modeling_baby import BabyTransformerForCausalLM


def register_auto() -> None:
    """Register the config/model with HF's Auto* factories (idempotent)."""
    try:
        AutoConfig.register(BabyTransformerConfig.model_type, BabyTransformerConfig)
        AutoModelForCausalLM.register(BabyTransformerConfig, BabyTransformerForCausalLM)
    except ValueError:
        # Already registered (e.g. module imported twice); nothing to do.
        pass


register_auto()

__all__ = [
    "BabyTransformerConfig",
    "BabyTransformerForCausalLM",
    "register_auto",
]
