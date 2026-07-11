# ETL image for SageMaker Processing.
# Slim on purpose: the char tokenizer + regex transforms are stdlib-only,
# so we install the nanogpt package WITHOUT its torch/numpy deps.
FROM python:3.12-slim

WORKDIR /app

COPY libs/nanogpt ./libs/nanogpt
RUN pip install --no-cache-dir --no-deps ./libs/nanogpt

COPY services/etl/prepare.py .

# SageMaker Processing mounts S3 data at these paths.
ENTRYPOINT ["python", "prepare.py", \
            "--input-dir", "/opt/ml/processing/input", \
            "--output-dir", "/opt/ml/processing/output"]
