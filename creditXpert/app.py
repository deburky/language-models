"""app.py."""

from flask import Flask, request, jsonify
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import SQLiteVSS
import openai
import os
import json

app = Flask(__name__)

# Load the embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Load SQLite-VSS database
DB_FILE = "db/vss.db"
TABLE_NAME = "document"


def get_vector_db():
    """Return an instance of SQLiteVSS connected to the database."""
    connection = SQLiteVSS.create_connection(db_file=DB_FILE)
    return SQLiteVSS(
        table=TABLE_NAME, embedding=embedding_function, connection=connection
    )


# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_response(query, context):
    """Generate a response using OpenAI's latest API."""
    if not openai.api_key:
        return "Error: OpenAI API key not found."

    messages = [
        {
            "role": "system",
            "content": "You are an expert in credit risk, finance, and AI. Answer concisely using the given context.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer in simple terms:",
        },
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        return response["choices"][0]["message"]["content"]
    except openai.error.InvalidRequestError as e:
        return f"Error: {e.message}"


@app.route("/query", methods=["POST"])
def similarity_search():
    """Handle similarity search and return an LLM-generated response."""
    data = request.json
    query = data.get("query")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        db = get_vector_db()
        retrieved_docs = db.similarity_search(query, k=3)

        # Ensure metadata is always a dictionary
        for doc in retrieved_docs:
            if isinstance(doc.metadata, str):  # If metadata is a string, convert it
                try:
                    doc.metadata = json.loads(doc.metadata)
                except json.JSONDecodeError:
                    doc.metadata = {}  # Set to empty dictionary if conversion fails

        if not retrieved_docs:
            return jsonify({"response": "No relevant information found."})

        # Combine top retrieved documents into a context string
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Generate final response
        answer = generate_response(query, context)

        return jsonify({"query": query, "answer": answer, "context_used": context})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
