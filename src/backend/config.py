"""
Centralized configuration for the Investment RAG Assistant.
"""

import os
import logging
from dotenv import load_dotenv

# ── Logging setup (applies to all backend modules) ────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Silence verbose libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

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
