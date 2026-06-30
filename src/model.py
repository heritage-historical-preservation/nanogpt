"""The transformer itself — the heart of the project.

Token + position embeddings -> N transformer blocks (causal self-attention +
MLP, each with residual + layernorm) -> final layernorm -> LM head.
"""

import torch.nn as nn


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        raise NotImplementedError("embeddings, blocks, ln_f, lm_head")

    def forward(self, idx, targets=None):
        raise NotImplementedError("return logits, loss")

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        raise NotImplementedError("autoregressive sampling loop")
