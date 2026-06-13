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
