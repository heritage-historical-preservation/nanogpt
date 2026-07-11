# PLAN — nanogpt-sacred: toy production deployment

## North star

Turn the hand-built char-level nanoGPT into a **toy with all the parts of a real
production ML system**: a browser prompt box hits a deployed model backend; a
`.txt` dropped in S3 flows through an ETL step into a training pipeline that can
be kicked off like a real one; everything wired with CI/CD and IaC on AWS.

The goal is pedagogical. Every component should be small enough to read in one
sitting, but shaped like its production counterpart.

## Locked decisions

| Decision | Choice | Why |
|---|---|---|
| ML stack | **Hybrid** — SageMaker Training Jobs (spot) for training, DIY container for serving | Managed training mirrors industry practice; DIY serving keeps the inference path fully visible |
| IaC | **AWS SAM / CloudFormation** | Lightest ceremony for a serverless-heavy stack |
| Cost posture | **Scale-to-zero** | Near-$0 idle; cold starts are an accepted tradeoff |
| Serving shape | **Lambda container image** behind API Gateway | The only DIY option that is both scale-to-zero and SAM-native (torch + checkpoint fit in the 10 GB image limit) |
| Sequencing | **Cloud-first** | Deploy a thin walking skeleton early; thicken it slice by slice |
| CI/CD | GitHub Actions → ECR → `sam deploy`, auth via **GitHub OIDC → IAM role** | No long-lived AWS keys in CI |
| Region | `us-east-1` (default; single-region) | Cheapest/most complete service coverage |
| Naming prefix | `nanogpt-sacred-brandon` | S3 bucket names are global; prefix everything |
| Budget guardrail | **AWS Budgets alert at $10/month** | Runaway-job protection, set up in Phase 1 |

## Target architecture

```
                        ┌──────────────────────────────────────────────┐
 browser ──► CloudFront │ S3 static frontend (prompt box)              │
                        └──────────────┬───────────────────────────────┘
                                       ▼
                            API Gateway ──► Lambda container (serving)
                                                 │ loads checkpoint
                                                 ▼
   you drop .txt          S3 ─────────────────────────────────────────
   ─────────────►  raw/  ──S3 event──► Lambda ETL (clean/transform)
                                            │
                                            ▼
                   processed/  ◄────────────┘        artifacts/ (versioned checkpoints)
                        │                                 ▲
                        └──► Step Functions ──► SageMaker Training Job (spot)
                             (drop → ETL → train → register)

   GitHub Actions ──build──► ECR ──deploy──► SAM/CloudFormation stacks
```

## Component map (local → cloud)

| Local piece | Cloud service |
|---|---|
| `data/raw/source.txt` | S3 `raw/` prefix |
| `prepare.py` transforms | Lambda ETL (S3-event triggered) |
| `data/processed/` + `meta.json` | S3 `processed/` prefix |
| `train.py` | SageMaker Training Job container |
| `checkpoints/model_best.pt` | S3 `artifacts/` (versioned) |
| `generate.py` | Lambda container + API Gateway `/generate` |
| — | S3 + CloudFront frontend |
| — | Step Functions orchestration |
| git repo | GitHub Actions CI/CD |

## Phases

### Phase 0 — Monorepo restructure (local, no AWS)
Extract the shared model core into an installable `libs/nanogpt` package;
split entry points into `services/{etl,training,serving}`.
**Done when:** `uv sync` installs the workspace and training/generation run
from their new homes.

### Phase 1 — Cloud foundation
AWS account wiring: `aws configure`, base SAM stack (S3 bucket with
`raw/`/`processed/`/`artifacts/` prefixes, ECR repo, OIDC role for GitHub
Actions), AWS Budgets alert at $10/mo. CI workflow that validates + deploys the
stack on push.
**Done when:** `git push` deploys the base stack; budget alert email confirmed.

### Phase 2 — ETL walking skeleton
`clean()` wrapped in a Lambda handler; S3 `raw/` PUT event triggers it; output
lands in `processed/` with `meta.json`. This slice proves the whole toolchain
(SAM, packaging, events, IAM, CI).
**Done when:** dropping a `.txt` into `raw/` produces cleaned output in
`processed/` with no manual steps.

### Phase 3 — Training pipeline
`train.py` in a SageMaker-compatible container (spot instances); reads
`processed/`, writes a versioned checkpoint + tokenizer meta to `artifacts/`.
Kick off via a CLI/console call first; Step Functions wiring comes in Phase 6.
**Done when:** a cloud training run produces a checkpoint in `artifacts/` that
generates coherent text locally.

### Phase 4 — Serving
Inference Lambda (container image): load checkpoint from `artifacts/` at cold
start, expose `/generate` through API Gateway with prompt/temperature/token
params.
**Done when:** `curl` against the public endpoint returns generated text.

### Phase 5 — Frontend
Static prompt UI on S3 + CloudFront calling the API.
**Done when:** the browser box round-trips a prompt to the deployed model.

### Phase 6 — Orchestration + observability
Step Functions chain (raw drop → ETL → train → publish artifact), CloudWatch
dashboards/alarms, structured logs, cost tags.
**Done when:** one S3 drop rolls a new model all the way to the endpoint.

## Prerequisites checklist

- [x] AWS auth: SSO profile `485161492380_AdministratorAccess` (BrandonAdmin) is
      the default via `AWS_PROFILE` in `~/.zshenv`; root login session removed.
      Refresh expired creds from the access portal (or set up `aws configure sso`).
- [x] Region confirmed: `us-east-1`
- [x] Naming prefix: `nanogpt-sacred-brandon`
- [x] Budget alert created at $10/month (80% actual + 100% forecast → email)
- [x] GitHub repo exists: `heritage-historical-preservation/nanogpt`

## Deployed resources (foundation stack: `nanogpt-sacred-foundation`)

| Resource | Value |
|---|---|
| Data bucket | `nanogpt-sacred-brandon-data` (zones: `raw/`, `processed/`, `artifacts/`; versioned) |
| ECR repo | `485161492380.dkr.ecr.us-east-1.amazonaws.com/nanogpt-sacred-brandon` |
| CI deploy role | `arn:aws:iam::485161492380:role/nanogpt-sacred-brandon-github-actions` (OIDC, main-branch only) |

## Status log

- 2026-06-30 — Phase-less beginnings: local model built and trained end-to-end.
- 2026-07-11 — Deployment direction + stack decided; this plan written; Phase 0
  (monorepo restructure) and Phase 1 (cloud foundation: bucket, ECR, OIDC role,
  budget, CI workflow) completed. Next: Phase 2, the ETL walking skeleton.
