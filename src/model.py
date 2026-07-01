# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

import torch.nn.functional as F

class Head(nn.Module):
    """One self-attention head."""

    def __init__(self, cfg, head_size):
        super().__init__()
        # Three projections: each token's vector -> query, key, value.
        # No bias, following nanoGPT convention.
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.key   = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)

        # The causal mask: a lower-triangular matrix of 1s, stored as a
        # buffer (not a parameter — it's fixed, never trained).
        self.register_buffer(
            "tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape          # C = n_embd
        q = self.query(x)          # (B, T, head_size)
        k = self.key(x)            # (B, T, head_size)
        v = self.value(x)          # (B, T, head_size)

        # Attention scores: every query dotted with every key.
        # (B,T,hs) @ (B,hs,T) -> (B,T,T). Scale by 1/sqrt(head_size).
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # (B, T, T)

        # Causal mask: zero out the future by setting it to -inf
        # BEFORE softmax (so it becomes 0 after softmax).
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)                          # (B, T, T)
        wei = self.dropout(wei)

        # Weighted sum of values.
        out = wei @ v              # (B,T,T) @ (B,T,hs) -> (B, T, hs)
        return out
class MultiHeadAttention(nn.Module): # several heads in parallel
    pass
class FeedForward(nn.Module):        # the per-token MLP
    pass
class Block(nn.Module):              # attention + FFN + residuals
    pass
class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Token embedding: a (vocab_size, n_embd) lookup table.
        # Row i is the learned vector for token ID i.
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)

        # Position embedding: a (block_size, n_embd) lookup table.
        # Row t is the learned vector for "being at position t".
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)

    def forward(self, idx):
        # idx: (B, T) tensor of token IDs.  B=batch, T=time/block_size.
        B, T = idx.shape

        # Look up token vectors: (B, T) ints -> (B, T, n_embd) vectors.
        tok_emb = self.token_embedding(idx)                       # (B, T, C)

        # Look up position vectors for positions 0..T-1: (T, n_embd).
        pos = torch.arange(T, device=idx.device)                  # (T,)
        pos_emb = self.position_embedding(pos)                    # (T, C)

        # Add them. Broadcasting handles (B,T,C) + (T,C).
        x = tok_emb + pos_emb                                     # (B, T, C)
        return x
    

if __name__ == "__main__":
    from src.dataset import Dataset

    ds = Dataset(config)              # sets config.vocab_size = 83
    assert config.vocab_size > 0, "vocab_size wasn't set — dataset must run before model"
    model = GPT(config).to(config.device)

    xb, yb = ds.get_batch("train")    # (B, T) integer IDs
    out = model(xb)                   # forward pass

    print(f"input  shape: {tuple(xb.shape)}  dtype: {xb.dtype}")   # (64,256) int64
    print(f"output shape: {tuple(out.shape)} dtype: {out.dtype}")  # (64,256,384) float
    print(f"device: {out.device}")

    # Parameter count so far (just the two embedding tables):
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")