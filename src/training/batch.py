"""Batch sampling for next-token training over a flat token array.

Both helpers operate on a 1-D numpy array of token ids (optionally mmap'd) and
sample random fixed-length windows; they are model- and corpus-agnostic.
"""

import numpy as np
import torch


def get_batch(
    data: np.ndarray, block_size: int, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a whole batch of random windows with one vectorized gather.

    Avoids the per-sample Python overhead of Dataset/DataLoader (64 __getitem__
    calls + collate per step). Works on the mmap'd token cache as well.

    Tokens are gathered and transferred as int32 (half the bytes of int64) and
    cast to long on the device, where nn.Embedding and cross-entropy need them.
    Used for the CPU path and validation; the training loop uses Prefetcher to
    overlap this work with GPU compute.
    """
    ix = np.random.randint(0, len(data) - block_size, size=batch_size)
    batch = torch.from_numpy(data[ix[:, None] + np.arange(block_size + 1)])
    x, y = batch[:, :-1], batch[:, 1:]
    if device.type == "cuda":
        # Pinned staging buffers let the H2D copy overlap with GPU compute; the
        # int32 -> int64 widening happens on the GPU after the smaller transfer.
        x = x.pin_memory().to(device, non_blocking=True).long()
        y = y.pin_memory().to(device, non_blocking=True).long()
        return x, y
    return x.long(), y.long()


def get_mntp_batch(
    data: np.ndarray,
    block_size: int,
    batch_size: int,
    device: torch.device,
    *,
    mask_id: int,
    vocab_size: int,
    n_special: int = 5,
    mask_prob: float = 0.15,
    rand_frac: float = 0.1,
    keep_frac: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch for masked-next-token prediction (the GPT-BERT BERT path).

    Returns ``(x, labels)`` where ``x`` is a mask-corrupted input window and
    ``labels`` holds, at each *output* position, the original token to predict
    (or ``-100`` to ignore). Output position ``i`` predicts the token at input
    position ``i+1`` -- exactly the offset the causal objective uses -- so the
    same output head serves both. A target at output ``i`` is created by
    corrupting input position ``i+1`` (BERT 80% ``<mask>`` / 10% random /
    10% unchanged) and recording the original token in ``labels[i]``.

    ``x`` is fed to ``GPTBERT(x, is_causal=False)`` and ``labels`` to a
    cross-entropy with ``ignore_index=-100``. Random replacements are drawn from
    ``[n_special, vocab_size)`` so they never land on a reserved special token.
    """
    ix = np.random.randint(0, len(data) - block_size, size=batch_size)
    window = torch.from_numpy(data[ix[:, None] + np.arange(block_size + 1)]).long()
    x = window[:, :-1].clone()   # (B, T) input positions 0..T-1
    nxt = window[:, 1:]          # (B, T) the token each output position predicts

    # Select which output positions to train. The last position has no in-window
    # successor to corrupt, so it is never selected.
    sel = torch.rand(batch_size, block_size) < mask_prob
    sel[:, -1] = False
    labels = torch.full_like(x, -100)
    labels[sel] = nxt[sel]

    # Corrupt the predicted token: output i reads position i+1, so shift the
    # selection right by one to find the input cells to corrupt.
    corrupt = torch.zeros_like(x, dtype=torch.bool)
    corrupt[:, 1:] = sel[:, :-1]
    r = torch.rand(batch_size, block_size)
    do_mask = corrupt & (r < 1.0 - rand_frac - keep_frac)
    do_rand = corrupt & (r >= 1.0 - rand_frac - keep_frac) & (r < 1.0 - keep_frac)
    x[do_mask] = mask_id
    x[do_rand] = torch.randint(n_special, vocab_size, (int(do_rand.sum()),), dtype=x.dtype)
    # The remaining corrupted cells keep their original token (the 10% "unchanged").

    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        labels = labels.pin_memory().to(device, non_blocking=True)
        return x, labels
    return x, labels


class Prefetcher:
    """Overlaps batch preparation with GPU compute.

    While the GPU runs step N, the next batch's CPU gather + H2D copy run on a
    side stream, hiding the data-path latency that otherwise sits on the
    critical path before each forward. Tokens move as int32 (half the bytes of
    int64) and are cast to long on the GPU. PyTorch's caching host/device
    allocators recycle the pinned and device buffers across steps, so there is
    no per-step cudaHostAlloc; record_stream keeps that recycling safe.
    """

    def __init__(
        self, data: np.ndarray, block_size: int, batch_size: int, device: torch.device
    ) -> None:
        self.data = data
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.stream = torch.cuda.Stream()
        self._preload()

    def _preload(self) -> None:
        ix = np.random.randint(0, len(self.data) - self.block_size, size=self.batch_size)
        batch = torch.from_numpy(self.data[ix[:, None] + np.arange(self.block_size + 1)])
        x, y = batch[:, :-1].pin_memory(), batch[:, 1:].pin_memory()
        with torch.cuda.stream(self.stream):
            self.next_x = x.to(self.device, non_blocking=True).long()
            self.next_y = y.to(self.device, non_blocking=True).long()

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        torch.cuda.current_stream().wait_stream(self.stream)
        x, y = self.next_x, self.next_y
        # Mark the buffers as in use on the default stream so the allocator does
        # not recycle them while the current step is still reading them.
        x.record_stream(torch.cuda.current_stream())
        y.record_stream(torch.cuda.current_stream())
        self._preload()
        return x, y
