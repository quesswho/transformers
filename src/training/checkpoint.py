"""Checkpoint save/load helpers, independent of the concrete model class.

A checkpoint bundles the model weights, optimizer state, step counter, the
tokenizer, and the model config so training can resume and inference can rebuild
the architecture from scratch.
"""

import torch


def save_checkpoint(path, model, optimizer, step, tokenizer, config) -> None:
    # torch.compile wraps the model and prefixes state_dict keys with
    # "_orig_mod."; save the underlying module so checkpoints load into an
    # uncompiled model.
    model = getattr(model, "_orig_mod", model)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "tokenizer": tokenizer.to_dict(),
        "config": config.to_dict(),
    }, path)


def load_checkpoint(path: str, map_location=None) -> dict:
    """Load a checkpoint's raw dict (weights, optimizer, step, tokenizer, config)."""
    return torch.load(path, map_location=map_location, weights_only=False)


def restore_training_state(model, optimizer, ckpt: dict) -> int:
    """Restore model + optimizer state from a checkpoint and return the next step.

    The optimizer state is skipped (with a warning) when its parameter layout no
    longer matches the model, so a resumed run with an altered architecture
    restarts the optimizer instead of crashing. A KeyError covers the optimizer
    *kind* changing between runs (e.g. resuming a pure-AdamW checkpoint with the
    hybrid Muon optimizer, or vice versa), whose state_dicts have different keys.
    """
    model.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except (ValueError, KeyError):
            print("Warning: optimizer state skipped (parameter layout changed); optimizer restarted.\n")
    return ckpt["step"] + 1 if "step" in ckpt else 1
