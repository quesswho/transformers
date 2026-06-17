"""Hybrid Muon + AdamW optimizer construction.

Muon (Jordan et al., 2024; ``torch.optim.Muon``, added in PyTorch 2.9)
orthogonalizes the momentum update of a network's 2D *hidden* weight matrices,
which improves sample efficiency over AdamW -- the axis a fixed-budget setup like
BabyLM rewards. It is a 2D-only optimizer: embeddings, the LM head, biases, and
1-D norm gains must keep AdamW (the built-in has no auxiliary-Adam mode).

Running two optimizers complicates the training loop, the LR schedule, and the
checkpoint -- which all assume a single optimizer object. ``HybridOptimizer``
hides the pair behind one ``zero_grad``/``step``/``state_dict`` surface so the
loop and ``save_checkpoint``/``restore_training_state`` are untouched. The
scheduler is the exception: ``torch.optim.lr_scheduler`` ``isinstance``-checks
for a real ``Optimizer``, so we attach one ``LambdaLR`` per sub-optimizer and
fan ``step``/``get_last_lr`` across them via ``HybridScheduler``.
"""

import torch


class HybridOptimizer:
    """Drive several real optimizers as one.

    Not an ``Optimizer`` subclass (it owns no params of its own); attach LR
    schedulers to the individual ``.optimizers`` via :func:`build_scheduler`,
    not to this object.
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers

    @property
    def param_groups(self) -> list[dict]:
        return [g for opt in self.optimizers for g in opt.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()

    def state_dict(self) -> dict:
        return {"hybrid": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict: dict) -> None:
        # A KeyError here means the checkpoint was written by a different
        # optimizer layout (e.g. a pure-AdamW run); restore_training_state
        # catches it and restarts the optimizer rather than crashing.
        for opt, sd in zip(self.optimizers, state_dict["hybrid"]):
            opt.load_state_dict(sd)


class HybridScheduler:
    """Fan ``step``/``get_last_lr`` across one scheduler per sub-optimizer."""

    def __init__(self, schedulers: list) -> None:
        self.schedulers = schedulers

    def step(self) -> None:
        for sched in self.schedulers:
            sched.step()

    def get_last_lr(self) -> list[float]:
        return [lr for sched in self.schedulers for lr in sched.get_last_lr()]


def build_optimizer(
    model,
    *,
    muon: bool = True,
    muon_lr: float = 5e-4,
    adam_lr: float = 1e-3,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
):
    """Partition ``model``'s parameters and return an optimizer.

    With ``muon=True``, hidden 2D weight matrices go to ``torch.optim.Muon`` and
    everything else (embeddings, the tied LM head, biases, norm gains) to AdamW,
    bundled in a :class:`HybridOptimizer`. A weight is Muon-eligible only if it is
    2-D *and* is neither an embedding nor the output projection -- the built-in's
    own example splits on ``ndim == 2`` alone, which would wrongly route the
    embedding (and, since it is tied, the LM head) through Muon. Deduping by
    ``id`` keeps the tied weight from being added twice.

    With ``muon=False`` every parameter falls through to a single fused AdamW,
    recovering the original behavior for an A/B comparison.
    """
    muon_params, adam_params, seen = [], [], set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        is_embed_or_head = ("embedding" in name) or ("projection" in name)
        if muon and p.dim() == 2 and not is_embed_or_head:
            muon_params.append(p)
        else:
            adam_params.append(p)

    if not muon_params:
        return torch.optim.AdamW(
            adam_params, lr=adam_lr, betas=betas, weight_decay=weight_decay, fused=True
        )

    muon_opt = torch.optim.Muon(
        muon_params, lr=muon_lr, weight_decay=weight_decay,
        # Scale each orthogonalized update to AdamW-comparable RMS so muon_lr is
        # width-independent and transfers across model sizes (Moonlight).
        adjust_lr_fn="match_rms_adamw",
    )
    adam_opt = torch.optim.AdamW(
        adam_params, lr=adam_lr, betas=betas, weight_decay=weight_decay, fused=True
    )
    return HybridOptimizer([muon_opt, adam_opt])


def build_scheduler(optimizer, lr_lambda):
    """A ``LambdaLR`` over ``optimizer``, transparently handling the hybrid case.

    The same ``lr_lambda`` multiplier drives every sub-optimizer, so the Muon and
    AdamW groups share the warmup + cosine *shape* while keeping their own peak
    LRs (each group's ``initial_lr``).
    """
    if isinstance(optimizer, HybridOptimizer):
        return HybridScheduler(
            [torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda) for opt in optimizer.optimizers]
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
