"""Query handler — embeds question and retrieves from a single backend."""

import json

from shared.aurora_backend import search as aurora_search
from shared.dynamo_backend import search as dynamo_search
from shared.embeddings import get_embedding


def lambda_handler(event: dict, context: object) -> dict:
    """Handle GET /query — retrieve top-k chunks from dynamo or aurora."""
    params = event.get("queryStringParameters") or {}
    backend = params.get("backend", "dynamo")
    user_id = params.get("user_id", "default")
    question = params.get("q")
    k = int(params.get("k", 5))

    if not question:
        return {"statusCode": 400, "body": json.dumps({"error": "q is required"})}

    embedding = get_embedding(question)
    results = (
        aurora_search(user_id, embedding, k)
        if backend == "aurora"
        else dynamo_search(user_id, embedding, k)
    )

    return {
        "statusCode": 200,
        "body": json.dumps({"backend": backend, "results": results}),
    }
