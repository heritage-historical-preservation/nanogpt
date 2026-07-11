# nanogpt-sacred

A small, from-scratch character-level GPT — built to understand the transformer
end to end, not to chase benchmarks.

## What it is

A minimal nanoGPT-style implementation:
- char-level tokenizer (build vocab, encode/decode)
- a single-file transformer (attention is the heart)
- a clean training loop and autoregressive sampler

Now growing into a **toy production deployment** on AWS — see `PLAN.md` for the
architecture and roadmap.

## Layout

```
libs/nanogpt/nanogpt/  # shared core (installable package)
  config.py            #   ALL hyperparameters in one place
  tokenizer.py         #   char-level: build vocab, encode/decode
  dataset.py           #   batching: sample sequences, make x/y pairs
  model.py             #   the transformer itself
services/
  etl/prepare.py       # clean raw text        -> future S3-triggered Lambda
  training/train.py    # training loop         -> future SageMaker job
  serving/generate.py  # autoregressive sampling -> future Lambda + API GW
data/
  raw/                 # source texts (gitignored)  ~ S3 raw/
  processed/           # cleaned corpus + meta.json ~ S3 processed/
checkpoints/           # saved weights (gitignored) ~ S3 artifacts/
infra/                 # SAM/CloudFormation templates (Phase 1+)
notebooks/             # exploration
```

## How to run

```bash
uv sync                                     # install the workspace
uv run python services/etl/prepare.py       # clean raw -> data/processed/
uv run python services/training/train.py    # train; best ckpt -> checkpoints/
uv run python services/serving/generate.py  # sample from a trained checkpoint
```

## Notes

_(your running notes go here)_
