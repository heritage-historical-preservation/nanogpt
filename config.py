"""ALL hyperparameters in one place.

Import this everywhere so there is a single source of truth for the run.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # --- data ---
    data_dir: str = "data/processed"
    train_split: float = 0.9

    # --- model ---
    block_size: int = 256        # context length (tokens of history)
    n_embd: int = 384            # embedding / residual stream width
    n_head: int = 6              # attention heads
    n_layer: int = 6             # transformer blocks
    dropout: float = 0.2
    vocab_size: int = 0          # filled in after building the tokenizer

    # --- training ---
    batch_size: int = 64
    max_iters: int = 5000
    eval_interval: int = 250
    eval_iters: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 1e-1
    grad_clip: float = 1.0

    # --- system ---
    device: str = "cpu"          # set to "cuda" / "mps" if available
    seed: int = 1337

    # --- io ---
    checkpoint_dir: str = "checkpoints"

    # --- sampling ---
    max_new_tokens: int = 500
    temperature: float = 1.0
    top_k: int = 200


config = Config()
