"""MiniLM embedding model — ONNX inference with S3 model cache in /tmp."""

import os

import boto3
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_BUCKET = os.environ["MODEL_BUCKET"]
MODEL_PREFIX = os.environ.get("MODEL_PREFIX", "models/minilm")
LOCAL_DIR = "/tmp/minilm"

_session: ort.InferenceSession | None = None
_tokenizer: Tokenizer | None = None


def _ensure_model():
    global _session, _tokenizer
    if _session is not None:
        return

    os.makedirs(LOCAL_DIR, exist_ok=True)
    s3 = boto3.client("s3")

    for filename in ("model.onnx", "tokenizer.json"):
        local_path = f"{LOCAL_DIR}/{filename}"
        if not os.path.exists(local_path):
            s3.download_file(MODEL_BUCKET, f"{MODEL_PREFIX}/{filename}", local_path)

    _session = ort.InferenceSession(f"{LOCAL_DIR}/model.onnx")
    _tokenizer = Tokenizer.from_file(f"{LOCAL_DIR}/tokenizer.json")
    _tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=128)
    _tokenizer.enable_truncation(max_length=128)


def _mean_pooling(
    token_embeddings: np.ndarray, attention_mask: np.ndarray
) -> np.ndarray:
    """Perform mean pooling on the token embeddings, taking into account the attention mask."""
    mask = attention_mask[:, :, np.newaxis].astype(np.float32)
    return np.sum(token_embeddings * mask, axis=1) / np.clip(
        mask.sum(axis=1), 1e-9, None
    )


def _normalize(v: np.ndarray) -> np.ndarray:
    """Normalize the embedding vectors to unit length."""
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(norm, 1e-9, None)


def get_embedding(text: str) -> list[float]:
    """Get the embedding for a given text using the MiniLM model."""
    _ensure_model()
    assert _session is not None and _tokenizer is not None
    encoding = _tokenizer.encode(text)

    input_ids = np.array([encoding.ids], dtype=np.int64)
    attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    input_names = {i.name for i in _session.get_inputs()}
    if "token_type_ids" in input_names:
        inputs["token_type_ids"] = np.zeros_like(input_ids)

    outputs = _session.run(None, inputs)
    embeddings = _mean_pooling(outputs[0], attention_mask)
    embeddings = _normalize(embeddings)
    return embeddings[0].tolist()
