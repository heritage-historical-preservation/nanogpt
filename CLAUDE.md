# CLAUDE.md

## What this is

A hand-built char-level nanoGPT being grown into a **toy production deployment**
on AWS — every part of a real ML system in miniature. The learning is the point:
prefer small, readable, production-*shaped* code over clever or exhaustive code.
See `PLAN.md` for the architecture, locked stack decisions, and phase roadmap —
read it before proposing infra or structural changes.

## Layout

```
libs/nanogpt/        # shared core: model, tokenizer, dataset, config (installable pkg)
services/etl/        # corpus cleaning/transforms  → becomes S3-triggered Lambda
services/training/   # training loop               → becomes SageMaker job container
services/serving/    # sampling/inference          → becomes Lambda container + API GW
data/raw|processed/  # local mirror of the S3 raw/ and processed/ zones (gitignored)
checkpoints/         # local mirror of S3 artifacts/ (gitignored)
infra/               # SAM/CloudFormation templates (from Phase 1)
notebooks/           # exploration
```

## Commands

```bash
uv sync                                    # install workspace (libs + root)
uv run python services/etl/prepare.py      # clean raw text -> data/processed/
uv run python services/training/train.py   # train; best checkpoint -> checkpoints/
uv run python services/serving/generate.py  # sample from a trained checkpoint
```

Run everything from the repo root — data paths are root-relative.

## Conventions

- **All hyperparameters live in `Config`** (`libs/nanogpt/nanogpt/config.py`).
  Never scatter magic numbers into services.
- `uv` for env/deps; the root `pyproject.toml` defines a uv workspace.
- Shared code goes in `libs/nanogpt`; services import `nanogpt.*` and stay thin.
- Tokenizer vocab is corpus-derived and saved as `meta.json` next to the
  processed corpus; serving must load it (never rebuild vocab at inference).
- `device` defaults to `mps` locally (Mac); cloud containers will be CPU/CUDA.
- Modules keep runnable `if __name__ == "__main__"` smoke-test blocks — preserve
  them when editing.
- Commit style: small atomic commits, imperative subject lines.

## AWS (from Phase 1)

- Region `us-east-1`; resource/naming prefix `nanogpt-sacred-brandon`.
- IaC is SAM/CloudFormation only (no Terraform/CDK). Serving is a Lambda
  container image — do not introduce Fargate/always-on endpoints.
- User runs interactive auth (`aws configure`); Claude runs `aws`/`sam` CLI
  after credentials exist. Budget cap $10/month — flag anything that could
  meaningfully bill.
