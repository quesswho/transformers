"""
Copy task: trains the transformer to reproduce its input sequence.
Demonstrates the full encoder-decoder pipeline on a single GPU with no
external data. Converges in a few hundred steps.

Run from project root:
    python examples/copy_task/copy_task.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
import torch.nn as nn

from transformer import EncoderDecoderTransformer

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
VOCAB_SIZE = 13  # 0=PAD, 1=SOS, 2=EOS, 3-12=data tokens


def generate_batch(
    batch_size: int, seq_len: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = torch.randint(3, VOCAB_SIZE, (batch_size, seq_len), device=device)
    sos = torch.full((batch_size, 1), SOS_IDX, device=device)
    eos = torch.full((batch_size, 1), EOS_IDX, device=device)
    src = data
    tgt_in = torch.cat([sos, data], dim=1)
    tgt_out = torch.cat([data, eos], dim=1)
    return src, tgt_in, tgt_out


def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}\n")

    BATCH_SIZE = 64
    SEQ_LEN = 8
    EPOCHS = 1000

    model = EncoderDecoderTransformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=128,
        nhead=4,
        num_layers=2,
        d_ff=256,
        dropout=0.1,
        max_len=50,
        pad_idx=PAD_IDX,
    ).to(device)

    counts = model.count_parameters()
    total = sum(counts.values())
    print(f"{'Component':<25} {'Params':>12}  {'%':>6}")
    for name, val in counts.items():
        print(f"  {name:<23} {val:>12,}  {val/total*100:>5.1f}%")
    print(f"  {'TOTAL':<23} {total:>12,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        src, tgt_in, tgt_out = generate_batch(BATCH_SIZE, SEQ_LEN, device)
        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch:3d}/{EPOCHS}  loss={loss.item():.4f}")

    print("\n--- Generation Demo ---")
    model.eval()
    correct = 0
    n = 10
    for _ in range(n):
        src, _, _ = generate_batch(1, SEQ_LEN, device)
        expected = src[0].tolist()
        predicted = model.generate(src, sos_idx=SOS_IDX, eos_idx=EOS_IDX)
        status = "OK  " if predicted == expected else "FAIL"
        print(f"  [{status}] expected={expected}  predicted={predicted}")
        correct += predicted == expected

    print(f"\nAccuracy: {correct}/{n}")


if __name__ == "__main__":
    train()
