# app.py

import uvicorn

from langserve import add_routes
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.chat_models import ChatOpenAI

from rag_chain import build_rag_chain
from fastapi import FastAPI
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


# Set response model
class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
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
        return {"query": query, "answer": answer, "context_used": context}
    except Exception as e:
        return ORJSONResponse(content={"error": str(e)}, status_code=500)


# LangServe RAG chain
rag_chain = build_rag_chain()
add_routes(app, rag_chain, path="/rag-chain")

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
