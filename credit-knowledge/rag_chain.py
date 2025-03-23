# rag_chain.py

import os
import json
import openai

from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import SQLiteVSS
from langchain_community.chat_models import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings


# Setup
DB_FILE = "db/vss.db"
TABLE_NAME = "document"
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_vector_db():
    """Get the SQLiteVSS instance."""
    connection = SQLiteVSS.create_connection(db_file=DB_FILE)
    return SQLiteVSS(
        table=TABLE_NAME, embedding=embedding_function, connection=connection
    )


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


def retrieve_docs(query: str):
    """Retrieve documents similar to the query."""
    db = get_vector_db()
    results = db.similarity_search(query, k=3)

    print(f"\n🔍 [RAG CHAIN] Retrieved {len(results)} docs for query: '{query}'")

    for doc in results:
        if isinstance(doc.metadata, str):
            try:
                doc.metadata = json.loads(doc.metadata)
            except json.JSONDecodeError:
                doc.metadata = {}
    return results




def build_rag_chain():
    retriever = RunnableLambda(
        lambda query: {"query": query, "docs": retrieve_docs(query)}
    )

    format_context = RunnableLambda(
        lambda inputs: {
            "query": inputs["query"],
            "context": "\n\n".join([doc.page_content for doc in inputs["docs"]]),
        }
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a credit risk expert AI. Always base your answer strictly on the provided context.",
            ),
            (
                "user",
                "Context:\n{context}\n\nQuestion: {query}\n\nAnswer in simple terms:",
            ),
        ]
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Final formatting to match FastAPI route
    format_final = RunnableLambda(
        lambda inputs: {
            "query": inputs["query"],
            "answer": inputs["answer"],
            "context_used": inputs["context"],
        }
    )

    return (
        retriever
        | format_context
        | RunnableMap(
            {
                "query": lambda x: x["query"],
                "answer": prompt | llm | StrOutputParser(),
                "context": lambda x: x["context"],
            }
        )
        | format_final
    )
