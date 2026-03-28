"""FastAPI app exposing RAG similarity search and LangServe routes."""

import uvicorn

from langserve import add_routes

from rag_chain import build_rag_chain
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

from rag_chain import retrieve_docs, generate_response

# FastAPI app config
app = FastAPI(
    title="LangChain Community API",
    description="API serving LangChain RAG and other chains",
    version="0.1.0",
    docs_url="/docs",
    default_response_class=ORJSONResponse,
)


class QueryRequest(BaseModel):
    """Body for the similarity-search endpoint."""

    query: str


class QueryResponse(BaseModel):
    """Structured answer plus the context string shown to the model."""

    query: str
    answer: str
    context_used: str


@app.post("/query")
async def similarity_search(request: QueryRequest) -> QueryResponse:
    """Handle similarity search and return an LLM-generated response."""
    query = request.query
    try:
        docs = retrieve_docs(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        answer = generate_response(query, context)
        return QueryResponse(
            query=query,
            answer=answer,
            context_used=context,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# LangServe RAG chain
rag_chain = build_rag_chain()
add_routes(app, rag_chain, path="/rag-chain")

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
