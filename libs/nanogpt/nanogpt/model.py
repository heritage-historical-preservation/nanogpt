# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanogpt.config import config

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
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        head_size = cfg.n_embd // cfg.n_head
        self.heads = nn.ModuleList([Head(cfg, head_size) for _ in range(cfg.n_head)])
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)


    def forward(self, x):
        # x; (B, T, n_embd)
        head_outputs = [head(x) for head in self.heads]
        return self.dropout(self.proj(torch.cat(head_outputs, dim=-1)))
    
class FeedForward(nn.Module):        # the per-token MLP
    """Per token MLP: 'think about what attention gathered.'"""

    def __init__(self,cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), # expand: 384 -> 1536
            nn.ReLU(),                              # nonlinearity
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),  # project back: 1536 -> 384
            nn.Dropout(cfg.dropout),
        )
    
    def forward(self, x):
        return self.net(x)                  # (B, T, C) -> (B, T, C), shape unchanged
class Block(nn.Module):              # attention + FFN + residuals
    def __init__(self, cfg):
        super().__init__()
        self.sa = MultiHeadAttention(cfg)
        self.ffwd = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)


    def forward(self, x):
        # x: (B, T, n_embd)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
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
        self.blocks = nn.Sequential(*(Block(cfg) for _ in range(cfg.n_layer)))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        # embedding space -> vocab logits
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def forward(self, idx, targets=None):
        # idx: (B, T) tensor of token IDs.  B=batch, T=time/block_size.
        B, T = idx.shape

        # Look up token vectors: (B, T) ints -> (B, T, n_embd) vectors.
        tok_emb = self.token_embedding(idx)                       # (B, T, C)

        # Look up position vectors for positions 0..T-1: (T, n_embd).
        pos = torch.arange(T, device=idx.device)                  # (T,)
        pos_emb = self.position_embedding(pos)                    # (T, C)

        # Add them. Broadcasting handles (B,T,C) + (T,C).
        x = tok_emb + pos_emb                                     # (B, T, C)

        # (B,T,C), refined through n_layer blocks
        x = self.blocks(x)
        # (B,T,C)
        x = self.ln_f(x)
        # (B,T,vocab_size)
        logits = self.lm_head(x)
        #if targets is None: loss = None else: compute cross_entropy between logits and targets
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))

        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        # idx: (B, T) — the running context of token IDs, grows by one each iteration
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx    

if __name__ == "__main__":
    from nanogpt.dataset import Dataset
    ds = Dataset(config)
    model = GPT(config).to(config.device)   # your embedding-only GPT

    xb, yb = ds.get_batch("train")
    logits, loss = model(xb, yb)
    print(f"logits: {tuple(logits.shape)}")
    print(f"loss: {loss.item():.4f}")
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
