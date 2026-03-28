"""Benchmark handler — fires both backends in parallel and returns latency comparison."""

import concurrent.futures
import json
import time

from shared.aurora_backend import search as aurora_search
from shared.dynamo_backend import search as dynamo_search
from shared.embeddings import get_embedding


def _timed(fn, *args):
    """Run fn(*args) and return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = fn(*args)
    return result, round((time.perf_counter() - t0) * 1000)


def lambda_handler(event: dict, context: object) -> dict:
    """Handle GET /benchmark — parallel retrieval from both backends with latency diff."""
    params = event.get("queryStringParameters") or {}
    user_id = params.get("user_id", "default")
    question = params.get("q")
    k = int(params.get("k", 5))

    if not question:
        return {"statusCode": 400, "body": json.dumps({"error": "q is required"})}

    embedding = get_embedding(question)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        dynamo_future = pool.submit(_timed, dynamo_search, user_id, embedding, k)
        aurora_future = pool.submit(_timed, aurora_search, user_id, embedding, k)
        dynamo_results, dynamo_ms = dynamo_future.result()
        aurora_results, aurora_ms = aurora_future.result()

    winner = "aurora" if aurora_ms < dynamo_ms else "dynamodb"

    return {
        "statusCode": 200,
        "body": json.dumps({
            "question": question,
            "latency_ms": {
                "dynamodb": dynamo_ms,
                "aurora": aurora_ms,
                "diff": abs(aurora_ms - dynamo_ms),
                "winner": winner,
            },
            "results": {
                "dynamodb": dynamo_results,
                "aurora": aurora_results,
            },
        }),
    }
