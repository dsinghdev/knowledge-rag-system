"""
Backend Service Layer for the Income Tax RAG Assistant.

Provides a clean interface (ask) between any frontend and the RAG pipeline.
In the future, this module can be replaced by HTTP calls to a FastAPI backend
without changing the frontend code.
"""

from vectorstore import get_vectorstore
from rag_chain import build_chain


# ── Module-level singleton ───────────────────────────────
_chain = None


def _get_chain():
    """Lazily initialise and cache the RAG chain."""
    global _chain
    if _chain is None:
        vectorstore = get_vectorstore()
        _chain = build_chain(vectorstore)
    return _chain


# ── Public API ───────────────────────────────────────────

def ask(question: str) -> dict:
    """
    Send a question to the RAG pipeline.

    Args:
        question: The user's natural-language question.

    Returns:
        dict with keys:
            - answer  (str):  The generated answer text.
            - sources (list[int]): Page numbers from source documents.
    """
    chain = _get_chain()
    result = chain.invoke({"query": question})

    sources = []
    if result.get("source_documents"):
        # Extract unique (Source Name, Page Number) pairs
        seen_sources = set()
        for doc in result["source_documents"]:
            name = doc.metadata.get("source_name", "Unknown Document")
            page = doc.metadata.get("page", 0) + 1  # Langchain is 0-indexed
            
            source_entry = f"{name} (Page {page})"
            if source_entry not in seen_sources:
                sources.append(source_entry)
                seen_sources.add(source_entry)

    return {
        "answer": result["result"],
        "sources": sources,
    }
