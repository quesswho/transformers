"""Reporting helpers: parameter breakdown table and training throughput."""

import time

import torch


def print_param_table(counts: dict[str, int]) -> None:
    """Print a model's parameter count broken down by component, with a total."""
    total = sum(counts.values())
    print(f"{'Component':<25} {'Params':>12}  {'%':>6}")
    for name, val in counts.items():
        print(f"  {name:<23} {val:>12,}  {val/total*100:>5.1f}%")
    print(f"  {'TOTAL':<23} {total:>12,}\n")


class ThroughputMeter:
    """Windowed tokens/sec measurement for the training loop.

    GPU work is async, so the clock is only read after a synchronize. The first
    window (warmup: cuDNN autotune, allocator growth, torch.compile) is excluded
    from the reported average, as is any window interrupted by validation.

    Usage per step::

        meter.record(x.numel())
        if logging:    tok_per_s, ms_per_step, warmup = meter.flush()
        if paused:     meter.reset_window()   # after logging/validation/ckpt

    The window must be reset after every pause so only pure training steps are
    timed.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.measured_time = 0.0
        self.measured_tokens = 0
        self.warmup = True
        self.reset_window()

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def reset_window(self) -> None:
        self._sync()
        self.window_start = time.perf_counter()
        self.window_tokens = 0
        self.window_steps = 0

    def record(self, tokens: int) -> None:
        self.window_tokens += tokens
        self.window_steps += 1

    def flush(self) -> tuple[float, float, bool]:
        """Close the current window; return (tok/s, ms/step, was_warmup).

        The warmup window is reported but excluded from the running average.
        """
        self._sync()
        dt = time.perf_counter() - self.window_start
        tok_per_s = self.window_tokens / dt
        ms_per_step = dt / self.window_steps * 1000
        was_warmup = self.warmup
        if self.warmup:
            self.warmup = False
        else:
            self.measured_time += dt
            self.measured_tokens += self.window_tokens
        return tok_per_s, ms_per_step, was_warmup

    def summary(self) -> str | None:
        """One-line average over all measured (non-warmup) windows, or None."""
        if self.measured_tokens == 0:
            return None
        return (f"Avg training speed: {self.measured_tokens / self.measured_time:,.0f} tok/s "
                f"({self.measured_tokens:,} tokens in {self.measured_time:.1f}s, "
                f"warmup & validation excluded)")
