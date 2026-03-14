"""
Centralized configuration for the Investment RAG Assistant.
"""

import os
import logging
from dotenv import load_dotenv

# ── Logging setup (applies to all backend modules) ────
from motifer import LogFactory

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Initialize Motifer LogFactory with mandatory service name
# This provides structured logging and pattern validation
factory = LogFactory(service="investment-rag-assistant", log_level=LOG_LEVEL)
# We use standard logging configuration but motifer helps with formatting and consistency
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# Silence verbose libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.INFO)

# Load environment variables from project root .env
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# ── Paths ──────────────────────────────────────────────
# config.py lives in src/backend/, so go up TWO levels to reach project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PDF_DIR = os.path.join(PROJECT_ROOT, "data")
INDEX_PATH = os.path.join(PROJECT_ROOT, "faiss_index")

# ── API ────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ── Models ─────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0

# ── Chunking ───────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
