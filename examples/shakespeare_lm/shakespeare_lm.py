"""
Shakespeare character-level language model.

Uses the Encoder stack as a decoder-only transformer (GPT-style):
passing a causal mask turns bidirectional self-attention into an
autoregressive LM with no changes to the architecture.

Run from project root:
    python examples/shakespeare_lm/shakespeare_lm.py
"""

import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformer import Encoder

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.join(os.path.dirname(__file__), "shakespeare.txt")


def download_data() -> str:
    if not os.path.exists(DATA_PATH):
        print("Downloading tinyshakespeare...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return f.read()


class CharDataset(Dataset):
    def __init__(self, data: list[int], block_size: int) -> None:
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class LanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        d_ff: int,
        dropout: float,
        block_size: int,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, nhead, num_layers, d_ff, dropout, block_size)
        self.head = nn.Linear(d_model, vocab_size)
        causal = torch.tril(torch.ones(block_size, block_size)).bool()
        self.register_buffer("causal_mask", causal.unsqueeze(0).unsqueeze(0))
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        mask = self.causal_mask[:, :, :T, :T]
        return self.head(self.encoder(x, mask))

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, steps: int, temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        block_size = self.causal_mask.size(-1)
        for _ in range(steps):
            ctx = idx[:, -block_size:]
            logits = self(ctx)[:, -1, :]
            probs = torch.softmax(logits / temperature, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}\n")

    BLOCK_SIZE = 256
    BATCH_SIZE = 64
    MAX_STEPS = 2000
    D_MODEL = 256
    NHEAD = 4
    NUM_LAYERS = 4
    D_FF = 1024
    DROPOUT = 0.1
    LR = 3e-4

    text = download_data()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = [stoi[c] for c in text]

    split = int(0.9 * len(data))
    train_dataset = CharDataset(data[:split], BLOCK_SIZE)
    val_dataset = CharDataset(data[split:], BLOCK_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    print(f"Vocab size: {vocab_size}  |  Train chars: {len(data[:split]):,}  |  Val chars: {len(data[split:]):,}\n")

    model = LanguageModel(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        block_size=BLOCK_SIZE,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_iter = iter(train_loader)
    for step in range(1, MAX_STEPS + 1):
        model.train()
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 500 == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    vlogits = model(vx)
                    val_losses.append(criterion(vlogits.view(-1, vocab_size), vy.view(-1)).item())
                    if len(val_losses) >= 20:
                        break
            val_loss = sum(val_losses) / len(val_losses)
            print(f"Step {step:5d}/{MAX_STEPS}  train={loss.item():.4f}  val={val_loss:.4f}")

    save_path = os.path.join(os.path.dirname(__file__), "shakespeare_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "config": {
            "vocab_size": vocab_size,
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_layers": NUM_LAYERS,
            "d_ff": D_FF,
            "dropout": DROPOUT,
            "block_size": BLOCK_SIZE,
        },
    }, save_path)
    print(f"\nModel saved → {save_path}")

    print("\n--- Generated text (temperature=0.8) ---\n")
    seed = "ROMEO:"
    ctx = torch.tensor([[stoi[c] for c in seed]], dtype=torch.long, device=device)
    out = model.generate(ctx, steps=500, temperature=0.8)
    generated = "".join(itos[i] for i in out[0].tolist())
    print(generated)


if __name__ == "__main__":
    train()
