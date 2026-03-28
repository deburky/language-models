"""Ingest handler — chunks text, embeds with MiniLM, dual-writes to DynamoDB and Aurora."""

import json
import os

import boto3

from shared.aurora_backend import write_chunks as aurora_write
from shared.dynamo_backend import write_chunks as dynamo_write
from shared.embeddings import get_embedding

_lambda_client = boto3.client("lambda")


def chunk_text(text: str, size: int = 256, overlap: int = 32) -> list[str]:
    """Split text into overlapping fixed-size chunks."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks


def lambda_handler(event: dict, context: object) -> dict:
    """Handle POST /ingest — return 202 immediately, process async via Event invocation."""
    # Async re-entry: event contains 'async_payload' key when invoked by ourselves
    if "async_payload" in event:
        _process(event["async_payload"])
        return {"statusCode": 200, "body": json.dumps({"status": "done"})}

    body = json.loads(event.get("body") or "{}")
    user_id = body.get("user_id", "default")
    document_id = body.get("document_id")
    text = body.get("text")

    if not document_id or not text:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "document_id and text are required"}),
        }

    # Fire-and-forget: invoke self with InvocationType=Event (async)
    _lambda_client.invoke(
        FunctionName=os.environ["AWS_LAMBDA_FUNCTION_NAME"],
        InvocationType="Event",
        Payload=json.dumps(
            {
                "async_payload": {
                    "user_id": user_id,
                    "document_id": document_id,
                    "text": text,
                }
            }
        ),
    )

    return {
        "statusCode": 202,
        "body": json.dumps(
            {
                "document_id": document_id,
                "status": "accepted",
            }
        ),
    }


def _process(payload: dict) -> None:
    """Chunk, embed, and dual-write a document to both backends."""
    user_id = payload["user_id"]
    document_id = payload["document_id"]
    text = payload["text"]

    chunks = chunk_text(text)
    embeddings = [get_embedding(chunk) for chunk in chunks]

    dynamo_write(user_id, document_id, chunks, embeddings)
    aurora_write(user_id, document_id, chunks, embeddings)
