"""Local LLM inference using Qwen2.5-0.5B-Instruct ONNX INT8 via onnxruntime."""

import os

import boto3
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_BUCKET = os.environ["MODEL_BUCKET"]
LLM_PREFIX = os.environ.get("LLM_PREFIX", "models/qwen")
LOCAL_DIR = "/tmp/qwen"
MAX_NEW_TOKENS = 64

# Qwen2.5 special token IDs
_EOS_IDS = {151643, 151645}  # <|endoftext|> and <|im_end|>

_session: ort.InferenceSession | None = None
_tokenizer: Tokenizer | None = None


def _ensure_model() -> None:
    """Download Qwen ONNX model and tokenizer from S3 on cold start."""
    global _session, _tokenizer
    if _session is not None:
        return

    os.makedirs(LOCAL_DIR, exist_ok=True)
    s3 = boto3.client("s3")

    for filename in ("model.onnx", "tokenizer.json"):
        local_path = f"{LOCAL_DIR}/{filename}"
        if not os.path.exists(local_path):
            s3.download_file(MODEL_BUCKET, f"{LLM_PREFIX}/{filename}", local_path)

    _session = ort.InferenceSession(
        f"{LOCAL_DIR}/model.onnx",
        providers=["CPUExecutionProvider"],
    )
    _tokenizer = Tokenizer.from_file(f"{LOCAL_DIR}/tokenizer.json")


def generate_answer(question: str, chunks: list[dict]) -> str:
    """Generate a grounded answer from retrieved chunks using greedy decoding."""
    _ensure_model()
    assert _session is not None and _tokenizer is not None

    context = "\n\n".join(
        f"[{c['document_id']}, chunk {c['chunk_index']}]\n{c['chunk_text']}"
        for c in chunks
    )
    prompt = (
        "<|im_start|>system\n"
        "Answer concisely based only on the context provided.<|im_end|>\n"
        "<|im_start|>user\n"
        f"Context:\n{context}\n\nQuestion: {question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    input_ids: list[int] = _tokenizer.encode(prompt).ids
    generated: list[int] = []

    for _ in range(MAX_NEW_TOKENS):
        ids = np.array([input_ids + generated], dtype=np.int64)
        mask = np.ones_like(ids, dtype=np.int64)
        pos_ids = np.arange(ids.shape[1], dtype=np.int64)[np.newaxis, :]
        logits = _session.run(None, {"input_ids": ids, "attention_mask": mask, "position_ids": pos_ids})[0]
        next_token = int(np.argmax(logits[0, -1, :]))
        if next_token in _EOS_IDS:
            break
        generated.append(next_token)

    return _tokenizer.decode(generated)
