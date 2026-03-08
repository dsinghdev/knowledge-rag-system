"""
Vector store management: Multi-format data loading, chunking, FAISS index creation & loading.
"""

import os
import glob
import logging

logger = logging.getLogger(__name__)

from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    PDF_DIR,
    INDEX_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def _get_embeddings():
    """Return the embedding model instance (runs locally, no API key needed)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
    )


def load_and_split_data():
    """Load data from various formats in the data/ folder and subfolders."""
    logger.info("Scanning directory: %s", PDF_DIR)
    
    # Load PDFs
    pdf_loader = DirectoryLoader(PDF_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader, silent_errors=True)
    # Load Text and Markdown
    text_loader = DirectoryLoader(PDF_DIR, glob="**/*.txt", loader_cls=TextLoader, silent_errors=True)
    md_loader = DirectoryLoader(PDF_DIR, glob="**/*.md", loader_cls=TextLoader, silent_errors=True)

    documents = []
    for loader in [pdf_loader, text_loader, md_loader]:
        docs = loader.load()
        # Add metadata source_name if missing
        for doc in docs:
            if "source_name" not in doc.metadata:
                doc.metadata["source_name"] = os.path.basename(doc.metadata.get("source", "Unknown"))
        documents.extend(docs)

    if not documents:
        raise FileNotFoundError(
            f"No valid data files found in {PDF_DIR}. "
            "Please add some investment documents (PDF, TXT, or MD)."
        )

    logger.info("Loaded %d document pages/files total.", len(documents))

    # Section-aware splitting logic
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\nSection ", "\nSection ", 
            "\n\nChapter ", "\nChapter ",
            "\n\n", "\n", " ", ""
        ],
        is_separator_regex=False
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split into %d chunks", len(chunks))
    return chunks


def create_vectorstore():
    """Build a FAISS index from documents and save to disk."""
    logger.info("Loading and splitting data...")
    chunks = load_and_split_data()

    logger.info("Creating embeddings & FAISS index...")
    embeddings = _get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)

    logger.info("Vector store saved to disk at %s", INDEX_PATH)
    return vectorstore


def load_vectorstore():
    """Load an existing FAISS index from disk."""
    embeddings = _get_embeddings()
    return FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_vectorstore():
    """Return the vectorstore — create it if it doesn't exist yet."""
    index_file = os.path.join(INDEX_PATH, "index.faiss")
    if os.path.exists(index_file):
        logger.info("Loading existing vector store from %s", INDEX_PATH)
        return load_vectorstore()
    else:
        logger.info("No existing index found — building from scratch...")
        return create_vectorstore()
