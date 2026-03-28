"""DynamoDB backend — vector storage with brute-force cosine similarity search."""

import os
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Key

from shared.similarity import cosine_similarity

_resource = boto3.resource("dynamodb")


def _table():
    """Return the DynamoDB table resource."""
    return _resource.Table(os.environ["DYNAMODB_TABLE"])


def _to_decimal(lst: list[float]) -> list[Decimal]:
    """Convert a list of floats to a list of Decimals for DynamoDB storage."""
    return [Decimal(str(v)) for v in lst]


def write_chunks(
    user_id: str, document_id: str, chunks: list[str], embeddings: list[list[float]]
):
    """Write document chunks and their embeddings to DynamoDB."""
    table = _table()
    with table.batch_writer() as batch:
        for i, (text, embedding) in enumerate(zip(chunks, embeddings)):
            batch.put_item(
                Item={
                    "user_id": user_id,
                    "document_chunk": f"{document_id}#{i}",
                    "document_id": document_id,
                    "chunk_index": i,
                    "chunk_text": text,
                    "embedding": _to_decimal(embedding),
                }
            )


def search(user_id: str, query_embedding: list[float], k: int = 5) -> list[dict]:
    """Search for the top-k most similar document chunks for a given user and query embedding."""
    table = _table()
    items = []

    resp = table.query(KeyConditionExpression=Key("user_id").eq(user_id))
    items.extend(resp["Items"])
    while "LastEvaluatedKey" in resp:
        resp = table.query(
            KeyConditionExpression=Key("user_id").eq(user_id),
            ExclusiveStartKey=resp["LastEvaluatedKey"],
        )
        items.extend(resp["Items"])

    scored = []
    for item in items:
        vec = [float(x) for x in item["embedding"]]
        score = cosine_similarity(query_embedding, vec)
        scored.append(
            {
                "chunk_text": item["chunk_text"],
                "document_id": item["document_id"],
                "chunk_index": int(item["chunk_index"]),
                "score": round(score, 4),
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]
