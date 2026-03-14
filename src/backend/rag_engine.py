"""
Backend Service Layer for the Investment RAG Assistant.

Provides a clean interface (ask) between any frontend and the RAG pipeline.
In the future, this module can be replaced by HTTP calls to a FastAPI backend
without changing the frontend code.
"""

import logging

from vectorstore import get_vectorstore
from rag_chain import build_chain
from config import factory

logger = factory.initialize()


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

import time

def ask(question: str) -> dict:
    """
    Send a question to the RAG pipeline.
    """
    start_time = time.time()
    
    # Fast-path for simple greetings
    cleaned_q = question.strip().lower().rstrip("?!.")
    if cleaned_q in GREETINGS:
        logger.info("Greeting detected ('%s') — bypassing RAG chain.", cleaned_q)
        return {
            "answer": (
                "Hello! 🧑‍💼 I am your **Knowledge RAG Assistant**.\n\n"
                "I'm here to help you understand your documents. "
                "How can I help you today?"
            ),
            "sources": [],
            "status": "success"
        }

    try:
        chain = _get_chain()
        logger.info("Processing query: %.80s...", question)
        
        retrieval_start = time.time()
        result = chain.invoke({"question": question})
        process_time = time.time() - retrieval_start
        
        sources = []
        if result.get("context"):
            # Filter out -1, convert to 1-indexed, and keep unique sorted integers
            sources = sorted({
                int(doc.metadata.get("page", -1)) + 1 
                for doc in result["context"] 
                if doc.metadata.get("page", -1) != -1
            })

        total_time = time.time() - start_time
        logger.info("Query answered in %.2fs (RAG process: %.2fs). %d source(s) found.", 
                    total_time, process_time, len(sources))
        
        return {
            "answer": result["answer"],
            "sources": sources,
            "status": "success",
            "metadata": {
                "process_time": round(process_time, 2),
                "total_time": round(total_time, 2)
            }
        }

    except Exception as e:
        error_msg = str(e)
        logger.error("Error processing query: %s", error_msg)
        
        # Categorize common errors for better frontend display
        error_type = "general_error"
        if "GOOGLE_API_KEY" in error_msg or "401" in error_msg:
            error_type = "api_key_error"
        elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            error_type = "connection_error"
        elif "not found" in error_msg.lower():
            error_type = "data_not_found"

        return {
            "answer": f"I'm sorry, I encountered an error while processing your request.",
            "sources": [],
            "status": "error",
            "error_type": error_type,
            "error_detail": error_msg
        }


from langchain_core.messages import HumanMessage, AIMessage

def ask_stream(question: str, chat_history: list = None):
    """
    Send a question to the RAG pipeline and yield chunks of the answer.
    Accepts an optional chat_history list of dicts: [{"role": "user"/"assistant", "content": "..."}]
    """
    start_time = time.time()
    
    # Fast-path for simple greetings
    cleaned_q = question.strip().lower().rstrip("?!.")
    if cleaned_q in GREETINGS:
        logger.info("Greeting detected ('%s') — bypassing RAG chain.", cleaned_q)
        greeting = (
            "Hello! 🧑‍💼 I am your **Knowledge RAG Assistant**.\n\n"
            "I'm here to help you understand your documents. "
            "How can I help you today?"
        )
        for char in greeting.split(" "):
            yield char + " "
        return

    try:
        chain = _get_chain()
        logger.info("Processing stream query: %.80s...", question)
        
        # Convert simple dict history to LangChain message objects
        lc_history = []
        if chat_history:
            for msg in chat_history:
                if msg.get("role") == "user":
                    lc_history.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    lc_history.append(AIMessage(content=msg.get("content", "")))
        
        retrieval_start = time.time()
        
        sources = []
        full_answer = ""
        
        # The LCEL chain yields dictionaries containing the intermediate steps
        for chunk in chain.stream({"input": question, "chat_history": lc_history}):
            if "context" in chunk and not sources:
                sources = sorted({
                    int(doc.metadata.get("page", -1)) + 1 
                    for doc in chunk["context"] 
                    if doc.metadata.get("page", -1) != -1
                })
            
            if "answer" in chunk:
                full_answer += chunk["answer"]
                yield chunk["answer"]
                
        process_time = time.time() - retrieval_start
        total_time = time.time() - start_time
        
        logger.info("Query stream answered in %.2fs (RAG process: %.2fs). %d source(s) found.", 
                    total_time, process_time, len(sources))
        
        # Finally, we want to yield a specific dict structure so the frontend knows what the sources are
        # But Streamlit write_stream just expects strings.
        # We'll just yield the sources as a specially formatted string at the end.
        if sources:
            yield f"\n\n[SOURCES_METADATA:{','.join(map(str, sources))}]"

    except Exception as e:
        error_msg = str(e)
        logger.error("Error processing query: %s", error_msg)
        
        error_type = "general_error"
        if "GOOGLE_API_KEY" in error_msg or "401" in error_msg:
            yield "🔑 **API Key Issue**\n\nThe Gemini API key appears to be invalid or missing. Please check your `.env` file."
        elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            yield "🌐 **Connection Error**\n\nI'm having trouble connecting to the AI service. Please check your internet connection."
        elif "not found" in error_msg.lower():
            yield "⚠️ **Data Not Found**\n\nI couldn't find any documents to search. Please ensure your PDFs are in the `data/` folder."
        else:
            yield f"❌ **An unexpected error occurred**\n\nDetail: `{error_msg}`"
