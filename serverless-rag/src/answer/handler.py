"""Answer handler — retrieves top-k chunks then generates a grounded answer locally."""

import json
import time

from shared.aurora_backend import search as aurora_search
from shared.dynamo_backend import search as dynamo_search
from shared.embeddings import get_embedding
from shared.llm import generate_answer


def lambda_handler(event: dict, context: object) -> dict:
    """Handle GET /answer — retrieve chunks then generate a local LLM answer."""
    params = event.get("queryStringParameters") or {}
    user_id: str = params.get("user_id", "default")
    question: str | None = params.get("q")
    backend: str = params.get("backend", "aurora")
    k: int = int(params.get("k", 5))

    if not question:
        return {"statusCode": 400, "body": json.dumps({"error": "q is required"})}

    embedding = get_embedding(question)

    t0 = time.perf_counter()
    chunks = (
        aurora_search(user_id, embedding, k)
        if backend == "aurora"
        else dynamo_search(user_id, embedding, k)
    )
    retrieval_ms = round((time.perf_counter() - t0) * 1000)

    t1 = time.perf_counter()
    answer = generate_answer(question, chunks)
    generation_ms = round((time.perf_counter() - t1) * 1000)

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "question": question,
                "answer": answer,
                "latency_ms": {
                    "retrieval": retrieval_ms,
                    "generation": generation_ms,
                    "total": retrieval_ms + generation_ms,
                },
                "backend": backend,
                "chunks_used": len(chunks),
            }
        ),
    }
