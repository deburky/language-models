# 🤹 Multi-Task Model API

The Multi-Task Model API project is a FastAPI-based service that enables question answering (QA) and text generation powered by DistilBERT and DistilGPT2 models.

It also incorporates DuckDB for efficient storage and retrieval of text embeddings, allowing for similarity-based context retrieval.

Run the app as follows:

```bash
uv run app.py
```

You can access the API through documentation at `http://
localhost:8000/docs`.

> Future improvements will include combining models into a single service for QA, generation, and retrieval.

---
## 📦 Key Features

* **Question Answering**: Retrieve the most relevant context from stored embeddings and generate accurate responses.
* **Text Generation**: Generate text based on a given prompt using a pretrained text generation model.
* **Embedding Storage & Retrieval**: Utilize DuckDB to store and retrieve vectorized text for similarity search.

## ✨ Components

* FastAPI – API framework for handling requests.
* Transformers (Hugging Face) – Model pipelines for text embedding, QA, and text generation.
* DuckDB – Lightweight database for storing and querying embeddings.
* Scikit-learn – Cosine similarity calculation for context retrieval.
* Uvicorn – ASGI server for running the FastAPI app.