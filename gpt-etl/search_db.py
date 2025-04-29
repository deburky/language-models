"""search_db.py."""

import chromadb
from embed_anything import EmbeddingModel, WhichModel

# Initialize ChromaDB
DB_PATH = "db"
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection(name="gpt_db")

# Load model
model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Bert, model_id="sentence-transformers/all-MiniLM-L6-v2"
)


def search_chroma_db(query, top_k=5):
    """Search the Chroma DB collection for the top_k most similar documents to the query."""
    print(f"🔍 Searching for: {query}")
    query_embedding = model.embed_query([query])[0].embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    for i, result in enumerate(results["documents"]):
        print(f"Result {i + 1}: {result}")


# Example search
search_chroma_db("Improving credit scoring")
