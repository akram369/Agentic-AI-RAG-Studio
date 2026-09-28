import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import PINECONE_INDEX_NAME, OPENAI_API_KEY, PINECONE_API_KEY
from src.graph import build_rag_graph

app = FastAPI(
    title="Agentic AI RAG API",
    description="Stateful RAG Chatbot powered by LangGraph, Pinecone, and OpenAI for the Agentic AI eBook.",
    version="1.0.0",
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global graph instance
rag_graph = None


def get_graph():
    """Lazily initialize and return the compiled LangGraph workflow."""
    global rag_graph
    if rag_graph is None:
        target_index = os.getenv("PINECONE_INDEX_NAME", PINECONE_INDEX_NAME)
        rag_graph = build_rag_graph(index_name=target_index)
    return rag_graph


class QueryRequest(BaseModel):
    query: str = Field(..., description="The user query to be answered using the eBook.", example="What is Agentic AI according to the eBook?")


class QueryResponse(BaseModel):
    final_answer: str = Field(..., description="The generated grounded answer.")
    retrieved_context: List[str] = Field(..., description="Context chunks retrieved from the document.")
    confidence_score: float = Field(..., description="Relevance / confidence score between 0.0 and 1.0.")


@app.get("/")
def root():
    """API health and metadata endpoint."""
    return {
        "status": "healthy",
        "service": "Agentic AI RAG API",
        "index": os.getenv("PINECONE_INDEX_NAME", PINECONE_INDEX_NAME),
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "online",
        "openai_configured": bool(OPENAI_API_KEY),
        "pinecone_configured": bool(PINECONE_API_KEY),
    }


@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """
    Execute stateful RAG graph to answer user query strictly based on document chunks.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    try:
        graph = get_graph()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize RAG graph: {str(e)}",
        )

    try:
        initial_state = {
            "question": request.query.strip(),
            "context": [],
            "answer": "",
            "score": 0.0,
        }
        result = graph.invoke(initial_state)

        return QueryResponse(
            final_answer=result["answer"],
            retrieved_context=result["context"],
            confidence_score=result["score"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
