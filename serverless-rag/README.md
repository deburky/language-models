# serverless-rag

Serverless RAG benchmark: DynamoDB (brute-force cosine) vs Aurora PostgreSQL (pgvector IVFFlat ANN index), deployed with AWS SAM.

## Architecture

```
POST /ingest    →  chunk → embed → write to DynamoDB + Aurora (async, 202)
GET  /query     →  retrieve from one backend (?backend=dynamo|aurora)
GET  /benchmark →  retrieve from both in parallel, return latency diff
GET  /answer    →  retrieve top-k chunks, generate answer with local LLM
```

Embedding: `all-MiniLM-L6-v2` via ONNX Runtime.
LLM: `Qwen2.5-0.5B-Instruct` INT8 quantized via ONNX Runtime.

## Prerequisites

- AWS CLI configured
- SAM CLI
- `uv` (for bootstrap model export)
- Python 3.12 in system PATH — SAM needs it for the Lambda runtime:

```bash
brew install python@3.12
```

Note: `onnxruntime` only ships `manylinux2014_x86_64` wheels for Python 3.12. Using Python 3.13 or ARM64 architecture causes SAM dependency resolution to fail. Lambda runs on x86_64, Python 3.12.

## Bootstrap

Run once before deploying. Creates the Aurora cluster, exports MiniLM and Qwen to ONNX, uploads models to S3, stores endpoints in SSM.

```bash
bash bootstrap.sh
```

The script is idempotent — re-running skips already-exported models and silently continues if the Aurora cluster already exists.

Schema and pgvector extension are created automatically by the Lambda on first ingest — no psql required.

## Deploy

First deploy (interactive, saves config to `samconfig.toml`):

```bash
make deploy-guided
```

When prompted: stack name `serverless-rag`, region `us-east-1`, allow IAM role creation `y`, auth warnings `y`, save config `y`.

Subsequent deploys:

```bash
make deploy
```

## Usage

```bash
make setup-local   # pull live endpoints into local/env_variables.json
make seed          # load seeds/*.txt into both backends
make bench Q="What is express configuration?"
make answer Q="What is express configuration?"
```

### Ingest a document manually

```bash
make ingest DOC=doc-001 TEXT="Amazon Aurora is..."
```

### Query a specific backend

```bash
make answer Q="What is Aurora?" BACKEND=aurora
make answer Q="What is Aurora?" BACKEND=dynamo
```

## Known issues

### onnxruntime on Lambda ARM64

`onnxruntime >= 1.18` crashes on Lambda ARM64 with:

```
onnxruntime::OnnxRuntimeException: Attempt to use DefaultLogger but none has been registered
```

And older versions have no `aarch64` wheels on PyPI. Use x86_64 (`Architectures: [x86_64]`) with Python 3.12.

### aws-psycopg2 on ARM64

`aws-psycopg2` only has x86_64 binaries. Use `psycopg2-binary>=2.9` which has proper manylinux aarch64/x86_64 wheels.

### Aurora express configuration and PAM auth

Aurora with `--with-express-configuration` enforces IAM/PAM authentication. Password auth via psql fails even for the master user. The Lambda uses IAM token auth (`generate_db_auth_token`) which works correctly.

## Tear down

```bash
make clean
```

Deletes the SAM stack, Aurora instances and cluster, S3 model bucket, and SSM parameters.
