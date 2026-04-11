# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "boto3",
#     "pandas",
#     "tqdm",
#     "datasets",
# ]
# ///
"""
Zero-Shot Classification with AWS Bedrock — Batch Inference (Claude Haiku)

Bedrock batch inference flow:
  1. For each row in a DataFrame, build a full Claude API request (prompt,
     parameters) and wrap it as {"recordId": ..., "modelInput": {...}}.
  2. Write these records as JSONL files (shards) and upload them to S3.
  3. Call `CreateModelInvocationJob`, pointing Bedrock at the S3 input prefix,
     an output prefix, and a model ID. Bedrock reads each JSONL line, sends
     the `modelInput` payload to the model, and writes the response back to
     the output prefix as .jsonl.out files.
  4. Poll `GetModelInvocationJob` until the job status is Completed/Failed.
  5. Download and parse .jsonl.out files from S3 into a DataFrame.

Note: `modelInput` is a standard Claude Messages API request, so it can carry
any content the API accepts: text, base64 images, multi-turn conversations,
system prompts, tool definitions. For image-heavy workloads, base64 encoding
bloats shard size, so plan for more shards to stay within the 1 GiB/file and
5 GiB/job limits.
"""

import io
import json
import time
from typing import Any

import boto3
import pandas as pd
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------------------------
MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

# Bedrock batch constraints
MAX_FILE_BYTES = 1 * 1024**3  # 1 GiB hard limit per file
TARGET_FILE_MB = 700  # aim well under cap
JOB_MAX_BYTES = 5 * 1024**3  # ~5 GiB per job prefix
JOB_PAD_BYTES = 128 * 1024**2  # headroom

INSTRUCTIONS = (
    "Classify the text into the given categories.\n"
    "Output exactly one JSON object with the fields specified below.\n"
    "If unsure, use 0.\n"
)


# -----------------------------------------------------------------------------------------------
# Model input
# -----------------------------------------------------------------------------------------------
def build_model_input(record_id: str, text: str, task_prompt: str) -> dict:
    """Build a single Bedrock API request payload.

    Each record becomes one JSONL line in the input shard. Bedrock batch
    replays the returned dict against the model as-is.

    Example output for record_id="0", text="What is Maximum Likelihood Estimation?"::

        {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Classify this question into exactly one category.\n
                                Categories: ml, programming, devops, tool_use,
                                math_science, chat\n\n...\n\n
                                Return ONLY the category name, nothing else.\n\n
                                Input:\nid=0 | What is Maximum Likelihood Estimation?"
                        }
                    ]
                }
            ],
            "max_tokens": 64,
            "temperature": 0.1,
            "top_p": 0.9
        }
    """
    prompt = f"{task_prompt}\n\nInput:\nid={record_id} | {text}"
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "max_tokens": 64,
        "temperature": 0.1,
        "top_p": 0.9,
    }


# -----------------------------------------------------------------------------------------------
# Sharding & upload
# -----------------------------------------------------------------------------------------------
def _estimate_bytes_per_record(df: pd.DataFrame, task_prompt: str, n: int = 200) -> int:
    sample = df[["id", "text"]].head(n)
    total = 0
    for rid, text in sample.itertuples(index=False, name=None):
        rec = {
            "recordId": str(rid),
            "modelInput": build_model_input(rid, text, task_prompt),
        }
        total += len((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
    return max(1, total // max(1, len(sample)))


def write_jsonl_shards(
    df: pd.DataFrame,
    bucket: str,
    base_prefix: str,
    task_prompt: str,
    region: str,
    basename: str = "batch-haiku",
    target_file_mb: int = TARGET_FILE_MB,
):
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    s3 = boto3.client("s3", region_name=region)
    avg_bytes = _estimate_bytes_per_record(df, task_prompt)
    target_bytes = min(MAX_FILE_BYTES - (16 * 1024**2), target_file_mb * 1024**2)
    rows_per_file = max(100, int(target_bytes // avg_bytes))

    max_rows = int((JOB_MAX_BYTES - JOB_PAD_BYTES) // avg_bytes)
    if len(df) > max_rows:
        print(
            f"[warn] Truncating {len(df):,} → {max_rows:,} rows to stay under ~5 GiB total."
        )
        df = df.iloc[:max_rows].copy()

    ts = int(time.time())
    prefix = base_prefix.rstrip("/")
    written = []

    print(
        f"~{avg_bytes} B/record, ~{rows_per_file} rows/file (~{(rows_per_file * avg_bytes) / 1024**2:.1f} MB)"
    )

    for i in tqdm(range(0, len(df), rows_per_file), desc="Writing shards", unit="file"):
        part = df.iloc[i : i + rows_per_file][["id", "text"]]
        key = f"{prefix}/{basename}-{ts}-part-{i // rows_per_file:05d}.jsonl"

        buf = io.StringIO()
        for rid, text in part.itertuples(index=False, name=None):
            rec = {
                "recordId": str(rid),
                "modelInput": build_model_input(rid, text, task_prompt),
            }
            buf.write(json.dumps(rec, ensure_ascii=False) + "\n")

        body = buf.getvalue().encode("utf-8")
        if len(body) >= MAX_FILE_BYTES:
            raise RuntimeError(f"Shard would exceed 1 GiB: {key} ({len(body)} bytes)")

        s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
        written.append(f"s3://{bucket}/{key}")

    input_prefix_uri = f"s3://{bucket}/{prefix}/"
    return input_prefix_uri, written


# -----------------------------------------------------------------------------------------------
# Job submission
# -----------------------------------------------------------------------------------------------
def submit_bedrock_batch(
    in_prefix_uri: str, out_prefix_uri: str, role_arn: str, region: str
) -> str:
    bedrock = boto3.client("bedrock", region_name=region)
    job_name = f"batch-haiku-{int(time.time())}"
    resp = bedrock.create_model_invocation_job(
        jobName=job_name,
        roleArn=role_arn,
        modelId=MODEL_ID,
        inputDataConfig={
            "s3InputDataConfig": {"s3Uri": in_prefix_uri, "s3InputFormat": "JSONL"}
        },
        outputDataConfig={"s3OutputDataConfig": {"s3Uri": out_prefix_uri}},
    )
    job_arn: str = resp["jobArn"]
    print("Job ARN:", job_arn)
    return job_arn


def wait_for_job(job_arn: str, region: str, poll: int = 20) -> dict[str, Any]:
    bedrock = boto3.client("bedrock", region_name=region)
    while True:
        j: dict[str, Any] = bedrock.get_model_invocation_job(jobIdentifier=job_arn)
        print(j["status"], j.get("message", ""))
        if j["status"] in ("Completed", "Failed", "Stopped"):
            return j
        time.sleep(poll)


# -----------------------------------------------------------------------------------------------
# Output parsing
# -----------------------------------------------------------------------------------------------
def _extract_row(rec: dict) -> dict:
    record_id = rec.get("recordId")
    output = rec.get("modelOutput", {})
    content = output.get("content", [])
    txt = "".join(part.get("text", "") for part in content if "text" in part).strip()

    try:
        obj = json.loads(txt)
    except json.JSONDecodeError:
        obj = {}

    usage = output.get("usage", {})
    return {
        "recordId": record_id,
        "raw_text": txt,
        "parsed": obj,
        "input_tokens": int(usage.get("input_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", 0)),
    }


def load_bedrock_batch_outputs(job_output_prefix: str, region: str) -> pd.DataFrame:
    assert job_output_prefix.startswith("s3://")
    bucket, _, prefix = job_output_prefix[5:].partition("/")
    s3 = boto3.client("s3", region_name=region)
    paginator = s3.get_paginator("list_objects_v2")
    rows: list = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".jsonl.out"):
                continue
            body = s3.get_object(Bucket=bucket, Key=key)["Body"]
            rows.extend(
                _extract_row(json.loads(line)) for line in body.iter_lines() if line
            )
    if not rows:
        return pd.DataFrame(
            columns=["recordId", "raw_text", "parsed", "input_tokens", "output_tokens"]
        )
    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["recordId"], keep="first")
        .reset_index(drop=True)
    )


# -----------------------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------------------
TASK_PROMPT = """Classify this question into exactly one category.
Categories: ml, programming, devops, tool_use, math_science, chat

- ml: machine learning, statistics, model training, evaluation metrics
- programming: Python, SQL, TypeScript, code patterns, algorithms, data structures
- devops: Docker, git, Makefile, bash scripting, CI/CD, infrastructure
- tool_use: file reads, grep, glob, bash commands on a specific repo/project
- math_science: math, physics, chemistry, history, general knowledge
- chat: vague, meta, or conversational

Return ONLY the category name, nothing else."""

if __name__ == "__main__":
    import argparse
    import re

    from datasets import load_dataset

    parser = argparse.ArgumentParser(description="Bedrock batch zero-shot classification")
    parser.add_argument("--bucket", required=True, help="S3 bucket for batch I/O")
    parser.add_argument("--role-arn", required=True, help="IAM role ARN for Bedrock batch")
    parser.add_argument("--region", default="eu-central-1", help="AWS region (default: eu-central-1)")
    args = parser.parse_args()

    def extract_q(text):
        user = re.search(r"user<\|message\|>(.*?)<\|end\|>", text, re.DOTALL)
        return user.group(1).strip() if user else ""

    ds = load_dataset("deburky/gpt-oss-claude-code")
    rows = []
    for i, row in enumerate(ds["train"]):
        q = extract_q(row["text"])
        if q:
            rows.append({"id": str(i), "text": q[:500]})

    df = pd.DataFrame(rows)
    print(f"{len(df)} questions loaded")

    in_prefix, shards = write_jsonl_shards(
        df,
        bucket=args.bucket,
        base_prefix="zero-shot/input",
        task_prompt=TASK_PROMPT,
        region=args.region,
    )
    print(f"Wrote {len(shards)} shard(s) to {in_prefix}")

    job_arn = submit_bedrock_batch(
        in_prefix_uri=in_prefix,
        out_prefix_uri=f"s3://{args.bucket}/zero-shot/output/",
        role_arn=args.role_arn,
        region=args.region,
    )
    print("\nJob submitted. Go drink beer.")
    print(
        f'Check status:\n  aws bedrock get-model-invocation-job --job-identifier "{job_arn}" --region {args.region} --query status'
    )
