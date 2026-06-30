# nanogpt-sacred

A small, from-scratch character-level GPT — built to understand the transformer
end to end, not to chase benchmarks.

## What it is

A minimal nanoGPT-style implementation:
- char-level tokenizer (build vocab, encode/decode)
- a single-file transformer (attention is the heart)
- a clean training loop and autoregressive sampler

## Layout

```
config.py            # ALL hyperparameters in one place
data/
  raw/               # downloaded source texts (gitignored)
  processed/         # cleaned, concatenated corpus
  prepare.py         # preprocess + tokenize -> train/val token arrays
src/
  tokenizer.py       # char-level: build vocab, encode/decode
  dataset.py         # batching: sample sequences, make x/y pairs
  model.py           # the transformer itself
  train.py           # forward, loss, backward, step
  generate.py        # autoregressive sampling from a trained model
checkpoints/         # saved weights (gitignored)
notebooks/
  explore.ipynb      # poke at attention weights, embeddings, etc.
```

## How to run

```bash
pip install -r requirements.txt
python data/prepare.py        # build train/val token arrays
python -m src.train           # train
python -m src.generate        # sample from a checkpoint
```

## Notes

_(your running notes go here)_
