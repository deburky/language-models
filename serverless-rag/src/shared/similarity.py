"""Pure Python cosine similarity for DynamoDB brute-force vector search."""

import math


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate the cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    return 0.0 if mag1 == 0.0 or mag2 == 0.0 else dot / (mag1 * mag2)
