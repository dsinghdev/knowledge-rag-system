"""
Centralized configuration for the Income Tax RAG Assistant.
"""

import os
from dotenv import load_dotenv

# Load environment variables from project root .env
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── Paths ──────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PDF_DIR = os.path.join(PROJECT_ROOT, "data")
INDEX_PATH = os.path.join(PROJECT_ROOT, "faiss_index")

# ── API ────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ── Models ─────────────────────────────────────────────
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-1.5-flash"
LLM_TEMPERATURE = 0

# ── Chunking ───────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
