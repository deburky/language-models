# -*- coding: utf-8 -*-
"""etl_to_db.py."""

import glob
import os

import chromadb
import embed_anything
from embed_anything import EmbeddingModel, TextEmbedConfig, WhichModel
from tqdm import tqdm

# SETTINGS
PROJECT_FOLDER = "conversations"
GLOB_PATTERNS = [
    "*.md",
    "*.py",
    "*.js",
    "*.ts",
    "*.java",
    "*.c",
    "*.cpp",
    "*.h",
    "*.html",
    "*.css",
    "*.txt",
]

# 1. Load model
model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Bert, model_id="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Define config
config = TextEmbedConfig(
    chunk_size=500,
    batch_size=32,
)

DB_PATH = "db"
os.makedirs(DB_PATH, exist_ok=True)

# 3. Initialize ChromaDB
# Remove existing database if it exists
if os.path.exists(DB_PATH):
    for root, dirs, files in os.walk(DB_PATH, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(DB_PATH)

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.create_collection(name="gpt_db")

# 4. Find matching files manually
file_paths = []
for pattern in GLOB_PATTERNS:
    file_paths.extend(
        glob.glob(os.path.join(PROJECT_FOLDER, "**", pattern), recursive=True)
    )

print(f"Found {len(file_paths)} files to embed...")

# 5. Embed each file manually
embedded_data = []
for path in tqdm(file_paths[:10], desc="Embedding files"):
    print(f"Embedding: {path}")
    data = embed_anything.embed_file(
        file_name=path,
        embedder=model,
    )
    embedded_data.extend(data)  # In case data is chunked into multiple pieces

# 6. Store embeddings
print("🧠 Inserting into vector database...")
for idx, item in enumerate(embedded_data):
    collection.add(
        documents=[item.text],
        embeddings=[item.embedding],
        metadatas=[
            {
                "file_path": item.metadata.get("file_path", ""),
                "chunk_id": item.metadata.get("chunk", 0),
            }
        ],
        ids=[f"doc-{idx}"],
    )

print(f"✅ Indexed {len(embedded_data)} code chunks!")
