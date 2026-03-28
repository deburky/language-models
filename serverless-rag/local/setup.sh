#!/bin/bash
# Populate local/env_variables.json from live AWS resources.
# Run after bootstrap.sh and sam deploy.

set -euo pipefail

STACK="${STACK:-serverless-rag}"
REGION="${AWS_REGION:-us-east-1}"
OUT="$(dirname "$0")/env_variables.json"

echo "==> Fetching Aurora endpoint from SSM..."
AURORA_ENDPOINT=$(aws ssm get-parameter \
  --name "/serverless-rag/aurora-endpoint" \
  --query 'Parameter.Value' \
  --output text \
  --region "$REGION")

echo "==> Fetching model bucket from SSM..."
MODEL_BUCKET=$(aws ssm get-parameter \
  --name "/serverless-rag/model-bucket" \
  --query 'Parameter.Value' \
  --output text \
  --region "$REGION")

echo "==> Fetching DynamoDB table name from CloudFormation..."
TABLE_NAME=$(aws cloudformation describe-stacks \
  --stack-name "$STACK" \
  --query 'Stacks[0].Outputs[?OutputKey==`VectorsTableName`].OutputValue' \
  --output text \
  --region "$REGION" 2>/dev/null || echo "${STACK}-vectors")

echo "==> Writing $OUT..."
python3 - <<EOF
import json

env = {}
for fn in ("IngestFunction", "QueryFunction", "BenchmarkFunction"):
    env[fn] = {
        "DYNAMODB_TABLE": "$TABLE_NAME",
        "AURORA_ENDPOINT": "$AURORA_ENDPOINT",
        "AURORA_DB_NAME": "postgres",
        "AURORA_DB_USER": "postgres",
        "MODEL_BUCKET": "$MODEL_BUCKET",
        "MODEL_PREFIX": "models/minilm",
        "AWS_REGION": "$REGION"
    }

with open("$OUT", "w") as f:
    json.dump(env, f, indent=2)
    f.write("\n")

print("  AURORA_ENDPOINT:", "$AURORA_ENDPOINT")
print("  DYNAMODB_TABLE: ", "$TABLE_NAME")
EOF

echo "Done. env_variables.json is ready for local invocation."
