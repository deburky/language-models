#!/bin/bash
# Bootstrap Aurora PostgreSQL with express configuration.
# Run once before `sam deploy`. Requires AWS CLI + psql (or use CloudShell fallback).
#
# Note: --with-express-configuration was announced 2026-03-25.
# Verify the exact flag name against: aws rds create-db-cluster help

set -euo pipefail

# Load .env if present (picks up HF_TOKEN, etc.)
ENV_FILE="$(dirname "$0")/../.env"
[ -f "$ENV_FILE" ] && set -a && source "$ENV_FILE" && set +a

CLUSTER_ID="${CLUSTER_ID:-serverless-rag-db}"
REGION="${AWS_REGION:-us-east-1}"
DB_NAME="postgres"
DB_USER="postgres"
SSM_PARAM="/serverless-rag/aurora-endpoint"
MODEL_PREFIX="models/minilm"
MODEL_SSM="/serverless-rag/model-bucket"

echo "==> Creating Aurora PostgreSQL cluster (express configuration)..."
aws rds create-db-cluster \
  --db-cluster-identifier "$CLUSTER_ID" \
  --engine aurora-postgresql \
  --region "$REGION" \
  --no-cli-pager \
  --with-express-configuration >/dev/null 2>&1 || echo "==> Cluster already exists, continuing..."


echo "==> Waiting for DB instance to become available (~60s)..."
aws rds wait db-instance-available \
  --filters "Name=db-cluster-id,Values=$CLUSTER_ID" \
  --region "$REGION"

echo "==> Fetching instance endpoint..."
INSTANCE_ENDPOINT=$(aws rds describe-db-instances \
  --filters "Name=db-cluster-id,Values=$CLUSTER_ID" \
  --query 'DBInstances[0].Endpoint.Address' \
  --output text \
  --region "$REGION")

echo "==> Endpoint: $INSTANCE_ENDPOINT"

echo "==> Storing endpoint in SSM Parameter Store..."
aws ssm put-parameter \
  --name "$SSM_PARAM" \
  --value "$INSTANCE_ENDPOINT" \
  --type "String" \
  --overwrite \
  --region "$REGION" \
  --no-cli-pager

echo "==> Schema will be auto-created by the Lambda on first ingest."

# MiniLM ONNX model
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
MODEL_BUCKET="serverless-rag-models-${ACCOUNT_ID}"

echo "==> Creating S3 model bucket: $MODEL_BUCKET..."
aws s3 mb "s3://$MODEL_BUCKET" --region "$REGION" 2>/dev/null || echo "    Bucket exists, continuing..."

if [ -f /tmp/minilm-onnx/model.onnx ] && [ -f /tmp/minilm-onnx/tokenizer.json ]; then
  echo "==> MiniLM already exported, skipping."
else
echo "==> Exporting all-MiniLM-L6-v2 to ONNX (one-time)..."
cat > /tmp/export_minilm.py << 'EOF'
import os
os.environ.get('HF_TOKEN') and __import__('huggingface_hub').login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)

from optimum.onnxruntime import ORTModelForFeatureExtraction
from tokenizers import Tokenizer

os.makedirs('/tmp/minilm-onnx', exist_ok=True)
print("Exporting model to ONNX...")
model = ORTModelForFeatureExtraction.from_pretrained(
    'sentence-transformers/all-MiniLM-L6-v2', export=True
)
model.save_pretrained('/tmp/minilm-onnx')
print("Saving tokenizer...")
tok = Tokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
tok.save('/tmp/minilm-onnx/tokenizer.json')
print("Done.")
EOF
# Use Python 3.12 — protobuf C extensions are incompatible with Python 3.14+
HF_TOKEN="${HF_TOKEN:-}" uv run --python 3.12 \
  --with "optimum[onnxruntime]" \
  --with tokenizers \
  python /tmp/export_minilm.py
fi

echo "==> Uploading MiniLM files to S3..."
aws s3 cp /tmp/minilm-onnx/model.onnx     "s3://$MODEL_BUCKET/$MODEL_PREFIX/model.onnx"
aws s3 cp /tmp/minilm-onnx/tokenizer.json "s3://$MODEL_BUCKET/$MODEL_PREFIX/tokenizer.json"

if [ -f /tmp/qwen-onnx-int8/model_quantized.onnx ] && [ -f /tmp/qwen-onnx-int8/tokenizer.json ]; then
  echo "==> Qwen already exported, skipping."
else
echo "==> Exporting Qwen2.5-0.5B-Instruct to ONNX INT8 (one-time)..."
cat > /tmp/export_qwen.py << 'EOF'
import os

os.environ.get('HF_TOKEN') and __import__('huggingface_hub').login(
    token=os.environ['HF_TOKEN'], add_to_git_credential=False
)

from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

os.makedirs('/tmp/qwen-onnx', exist_ok=True)
os.makedirs('/tmp/qwen-onnx-int8', exist_ok=True)

print("Exporting to ONNX (use_cache=False for simple greedy decoding)...")
model = ORTModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-0.5B-Instruct',
    export=True,
    use_cache=False,
)
model.save_pretrained('/tmp/qwen-onnx')

print("Quantizing to INT8 (AVX2/x86_64)...")
quantizer = ORTQuantizer.from_pretrained('/tmp/qwen-onnx')
qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
quantizer.quantize(save_dir='/tmp/qwen-onnx-int8', quantization_config=qconfig)

print("Saving tokenizer...")
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
tok.backend_tokenizer.save('/tmp/qwen-onnx-int8/tokenizer.json')
print("Done.")
EOF
# Use Python 3.12 — protobuf C extensions are incompatible with Python 3.14+
HF_TOKEN="${HF_TOKEN:-}" uv run --python 3.12 \
  --with "optimum[onnxruntime]" \
  --with transformers \
  python /tmp/export_qwen.py
fi

echo "==> Uploading Qwen ONNX INT8 to S3..."
aws s3 cp /tmp/qwen-onnx-int8/model_quantized.onnx \
    "s3://$MODEL_BUCKET/models/qwen/model.onnx"
aws s3 cp /tmp/qwen-onnx-int8/tokenizer.json \
    "s3://$MODEL_BUCKET/models/qwen/tokenizer.json"

echo "==> Storing model bucket name in SSM..."
aws ssm put-parameter \
  --name "$MODEL_SSM" \
  --value "$MODEL_BUCKET" \
  --type "String" \
  --overwrite \
  --region "$REGION" \
  --no-cli-pager

echo ""
echo "Done."
echo "  Cluster:  $CLUSTER_ID"
echo "  Endpoint: $INSTANCE_ENDPOINT"
echo "  SSM:      $SSM_PARAM"
echo ""
echo "Next: sam deploy --guided"
