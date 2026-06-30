"""Autoregressive sampling from a trained model.

Loads a checkpoint + tokenizer, seeds with an optional prompt, and streams
generated characters.
"""

from config import config


def main():
    raise NotImplementedError("load checkpoint, encode prompt, model.generate, decode")


if __name__ == "__main__":
    main()
