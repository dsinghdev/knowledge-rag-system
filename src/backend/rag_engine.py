"""
Backend Service Layer for the Investment RAG Assistant.

Provides a clean interface (ask) between any frontend and the RAG pipeline.
In the future, this module can be replaced by HTTP calls to a FastAPI backend
without changing the frontend code.
"""

import logging

from vectorstore import get_vectorstore
from rag_chain import build_chain

logger = logging.getLogger(__name__)


# ── Module-level singleton ───────────────────────────────
_chain = None


def _get_chain():
    """Lazily initialise and cache the RAG chain."""
    global _chain
    if _chain is None:
        logger.info("Initialising RAG chain (first request)...")
        vectorstore = get_vectorstore()
        _chain = build_chain(vectorstore)
        logger.info("RAG chain ready.")
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
    logger.info("Processing query: %.80s...", question)
    result = chain.invoke({"query": question})

    sources = []
    if result.get("source_documents"):
        sources = sorted(
            {doc.metadata.get("page", -1) for doc in result["source_documents"]}
        )

    logger.info("Query answered. %d source(s) found.", len(sources))
    return {
        "answer": result["result"],
        "sources": sources,
    }
