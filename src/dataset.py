import torch
from src.tokenizer import CharTokenizer
from pathlib import Path
from config import config


class Dataset:
    def __init__(self, cfg):
        self.cfg = cfg
        text = Path(cfg.data_dir, "sample.txt").read_text(encoding="utf-8")

        # Build tokenizer from the corpus, and backfill vocab_size.
        self.tokenizer = CharTokenizer.from_text(text)
        cfg.vocab_size = self.tokenizer.vocab_size   # the derived value

        # Encode the entire corpus to one long tensor of IDs.
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

        # Train/val split — a simple positional cut.
        n = int(cfg.train_split * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split: str):
        data = self.train_data if split == "train" else self.val_data
        # Pick batch_size random starting points.
        assert len(data) > self.cfg.block_size, (
            f"{split} split ({len(data)}) too small for block_size "
            f"{self.cfg.block_size}"
        )
        ix = torch.randint(len(data) - self.cfg.block_size, (self.cfg.batch_size,))
        # x = block_size chars from each start; y = same, shifted +1.
        x = torch.stack([data[i : i + self.cfg.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + self.cfg.block_size] for i in ix])
        return x.to(self.cfg.device), y.to(self.cfg.device)
    

if __name__ == "__main__":
    ds = Dataset(config)

    # 1. Confirm vocab_size derived correctly (should be 83, not 0).
    print(f"vocab_size (derived): {config.vocab_size}")

    # 2. Confirm the train/val split.
    print(f"train tokens: {len(ds.train_data)}")
    print(f"val tokens:   {len(ds.val_data)}")

    # 3. Pull a batch and check shapes.
    xb, yb = ds.get_batch("train")
    print(f"x shape: {tuple(xb.shape)}")   # (batch_size, block_size)
    print(f"y shape: {tuple(yb.shape)}")   # same
    print(f"device:  {xb.device}")

    # 4. THE KEY CHECK: y is x shifted by one.
    print("\nx[0, :10]:", xb[0, :10].tolist())
    print("y[0, :10]:", yb[0, :10].tolist())
    print("x[0, 1:11]:", xb[0, 1:11].tolist(), "  <- should equal y[0, :10]")

    # 5. Programmatic proof of the shift, and a human-readable decode.
    assert xb[0, 1:11].tolist() == yb[0, :10].tolist(), "shift-by-one broken!"
    print("\nshift-by-one verified ✓")
    print("x decoded:", repr(ds.tokenizer.decode(xb[0, :40].tolist())))
    print("y decoded:", repr(ds.tokenizer.decode(yb[0, :40].tolist())))