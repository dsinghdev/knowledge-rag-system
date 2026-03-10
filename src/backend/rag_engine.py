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

# ── Greeting Bypass ──────────────────────────────────────
GREETINGS = {
    "hi", "hello", "hey", "hola", "namaste", "good morning", "good evening",
    "hellp", "helo", "hello there", "hi there", "hy", "hlo"
}

def ask(question: str) -> dict:
    """
    Send a question to the RAG pipeline.
    """
    # Fast-path for simple greetings
    cleaned_q = question.strip().lower().rstrip("?!.")
    if cleaned_q in GREETINGS:
        logger.info("Greeting detected ('%s') — bypassing RAG chain.", cleaned_q)
        return {
            "answer": (
                "Hello! 🧑‍💼 I am your **Investment RAG Assistant**.\n\n"
                "I'm here to help you understand Indian Small Savings and Investment schemes "
                "(like PPF, NPS, SSY) using official government documents.\n\n"
                "How can I help you today?"
            ),
            "sources": [],
        }

    chain = _get_chain()
    logger.info("Processing query: %.80s...", question)
    result = chain.invoke({"query": question})

    sources = []
    if result.get("source_documents"):
        # Filter out -1, convert to 1-indexed, and keep unique sorted integers
        # We NO LONGER return source_name/filenames as per user request.
        sources = sorted({
            int(doc.metadata.get("page", -1)) + 1 
            for doc in result["source_documents"] 
            if doc.metadata.get("page", -1) != -1
        })

    logger.info("Query answered. %d source(s) found.", len(sources))
    return {
        "answer": result["result"],
        "sources": sources,
    }
