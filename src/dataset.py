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
    from src.dataset import Dataset
    ds = Dataset(config)
    model_emb = GPT(config).to(config.device)   # your embedding-only GPT
    head = Head(config, head_size=16).to(config.device)

    xb, _ = ds.get_batch("train")
    x = model_emb(xb)                # (B, T, C) embeddings
    out = head(x)                    # (B, T, head_size)
    print(f"input:  {tuple(x.shape)}")     # (64, 256, 384)
    print(f"output: {tuple(out.shape)}")   # (64, 256, 16)

    # THE VISUAL CHECK: look at one attention matrix.
    # Re-run the internals for a tiny slice to see the weights.
    import torch
    B, T, C = x.shape
    q, k = head.query(x), head.key(x)
    wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
    wei = wei.masked_fill(head.tril[:T, :T] == 0, float("-inf"))
    wei = torch.softmax(wei, dim=-1)
    print("\nAttention row for token 4 (first 8 columns):")
    print(wei[0, 4, :8].tolist())
    print("Token 4 attending to positions 5-7 (the future):")
    print(wei[0, 4, 5:8].tolist(), " <- should be all 0.0")