"""HuggingFace `PreTrainedModel` wrapper around `DecoderOnlyTransformer`.

The wrapper holds the repo's native model unchanged and exposes the HF causal-LM
contract: `forward(input_ids, attention_mask=None, labels=None)` returning a
`CausalLMOutputWithPast` with logits (and loss when labels are given). That
`forward` is what the BabyLM evaluation harness calls to read per-token
log-likelihoods for BLiMP / EWoK / GLUE scoring.

Padding note: the decoder-only stack is causal-only and ignores
`attention_mask`. Run the eval at batch_size=1 (no padding) until a padding mask
is threaded through `TransformerStack`.
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from transformer import DecoderOnlyTransformer

from .configuration_baby import BabyTransformerConfig


class BabyTransformerForCausalLM(PreTrainedModel):
    config_class = BabyTransformerConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = False

    def __init__(self, config: BabyTransformerConfig) -> None:
        super().__init__(config)
        self.transformer = DecoderOnlyTransformer(config.to_model_config())
        # When the inner model ties its output projection to the input embedding,
        # the two share one tensor. Declare the derived key so `save_pretrained`
        # treats it as an intentional tie (drops it on save, restores it via
        # `tie_weights()` on load) instead of erroring on a shared tensor.
        if config.tie_embeddings:
            self._tied_weights_keys = {
                "transformer.projection.weight": "transformer.stack.embedding.weight"
            }
        # HF post-init: runs weight tying (a no-op reinit, since the inner model
        # already initialised and tied its own weights in its constructor).
        self.post_init()

    # ------------------------------------------------------------------
    # Embedding accessors (used by HF weight tying / resize helpers)
    # ------------------------------------------------------------------

    def get_input_embeddings(self) -> nn.Module:
        return self.transformer.stack.embedding

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.transformer.stack.embedding = value

    def get_output_embeddings(self) -> nn.Module:
        return self.transformer.projection

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.transformer.projection = new_embeddings

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,  # accepted for API compat; ignored (causal-only)
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        logits = self.transformer(input_ids)

        loss = None
        if labels is not None:
            # Standard causal-LM shift: predict token t+1 from tokens <= t.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)


# Bundle this file + an `auto_map` entry into every `save_pretrained` checkpoint
# so `AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)` resolves
# the model in the eval harness's process. Mirrors the config registration.
BabyTransformerForCausalLM.register_for_auto_class("AutoModelForCausalLM")
