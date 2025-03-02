"""app.py."""

import os
import uvicorn
import duckdb
import numpy as np
from filelock import FileLock
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, GenerationConfig
from sklearn.metrics.pairwise import cosine_similarity

# Create FastAPI application
app = FastAPI(title="Multi-Task Model API")


class ModelConfig(BaseModel):
    qa_model_name: str
    tg_model_name: str

MODEL_CONFIG = ModelConfig(
    qa_model_name="distilbert-base-uncased-distilled-squad",
    tg_model_name="distilbert/distilgpt2",
)

# Configuration for the generation pipeline
GEN_CONFIG = GenerationConfig(
    max_length=100,
    temperature=0.5,
    top_k=10,
    top_p=0.90,
    do_sample=True,
    eos_token_id=50256,
    pad_token_id=50256,
    repetition_penalty=2.0,
)

# Function to load the models
def load_models_from_config(model_config: ModelConfig):
    # Initialize QA pipeline
    qa_pipeline = pipeline(
        "question-answering",
        model=f"./models/{model_config.qa_model_name}",
        tokenizer=f"./models/{model_config.qa_model_name}",
    )

    # Initialize text-generation pipeline
    tg_pipeline = pipeline(
        "text-generation",
        model=f"./models/{model_config.tg_model_name}",
        tokenizer=f"./models/{model_config.tg_model_name}",
    )

    return qa_pipeline, tg_pipeline


# Load models from the configuration
qa_pipeline, tg_pipeline = load_models_from_config(MODEL_CONFIG)

# Function to generate embeddings using the QA model (distilbert)
# Initialize embedding pipeline once
embedding_pipeline = pipeline(
    "feature-extraction",
    model="distilbert-base-uncased-distilled-squad",
    tokenizer="distilbert-base-uncased-distilled-squad",
)


def generate_embeddings(text: str):
    embeddings = embedding_pipeline(text)
    return np.mean(embeddings[0], axis=0)


# Function to get most similar context based on embeddings from DuckDB
def get_most_similar_context(query: str):
    query_embedding = generate_embeddings(query).tolist()

    with duckdb.connect("db/embeddings.db") as con:
        result = con.execute(
            """
            SELECT text FROM embeddings 
            ORDER BY embedding <-> ? LIMIT 5
        """,
            [query_embedding],
        ).fetchone()

    return result[0] if result else ""


# Function to insert new embeddings into DuckDB
def insert_embedding(text: str, embedding: np.ndarray):
    lock = FileLock("db/embeddings.lock")  # Lock the DB for writing
    with lock:
        with duckdb.connect("db/embeddings.db") as con:
            con.execute(
                "INSERT INTO embeddings (text, embedding) VALUES (?, ?)",
                (text, embedding.tolist()),
            )


# Question Answering Endpoint
class QARequest(BaseModel):
    question: str


class QAResponse(BaseModel):
    answer: str


@app.post("/qa", response_model=QAResponse)
def answer_question(request: QARequest):
    try:
        # Find the most similar context from DuckDB
        context = get_most_similar_context(request.question)

        # Use the QA model to get the answer
        result = qa_pipeline(
            question=request.question,
            context=context,
            max_length=100,
        )
        return QAResponse(answer=result["answer"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# Text Generation Endpoint
class GenerationRequest(BaseModel):
    prompt: str


class GenerationResponse(BaseModel):
    generated_text: str


@app.post("/generate", response_model=GenerationResponse)
def generate_text(request: GenerationRequest):
    try:
        # Generate text with a maximum length of 50 tokens.
        result = tg_pipeline(
            request.prompt,
            generation_config=GEN_CONFIG
        )
        generated_text = result[0]["generated_text"]
        cleaned_text = generated_text.replace("\n", "").strip()
        return GenerationResponse(generated_text=cleaned_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/")
def read_root():
    return {"message": "Welcome to the Multi-Task Model API!"}


if __name__ == "__main__":
    uvicorn.run("app:app", reload=True)
