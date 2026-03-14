"""
Vector store management: Multi-format data loading, chunking, FAISS index creation & loading.
"""

import os
import glob
import logging

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
    factory
)

logger = factory.initialize()


def _get_embeddings():
    """Return the embedding model instance (runs locally, no API key needed)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
    )


def load_and_split_data():
    """Load data from various formats in the data/ folder and subfolders."""
    if not os.path.exists(PDF_DIR):
        logger.error("Data directory not found: %s", PDF_DIR)
        raise FileNotFoundError(f"Data directory '{PDF_DIR}' does not exist.")

    logger.info("Scanning directory for documents: %s", PDF_DIR)
    
    try:
        # Load PDFs
        pdf_loader = DirectoryLoader(PDF_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader, silent_errors=True)
        # Load Text and Markdown
        text_loader = DirectoryLoader(PDF_DIR, glob="**/*.txt", loader_cls=TextLoader, silent_errors=True)
        md_loader = DirectoryLoader(PDF_DIR, glob="**/*.md", loader_cls=TextLoader, silent_errors=True)

        documents = []
        for name, loader in [("PDF", pdf_loader), ("Text", text_loader), ("Markdown", md_loader)]:
            logger.debug("Loading %s files...", name)
            docs = loader.load()
            if docs:
                logger.info("  Successfully loaded %d %s document(s).", len(docs), name)
                # Add metadata source_name if missing
                for doc in docs:
                    if "source_name" not in doc.metadata:
                        doc.metadata["source_name"] = os.path.basename(doc.metadata.get("source", "Unknown"))
                documents.extend(docs)

        if not documents:
            logger.warning("No documents found in %s", PDF_DIR)
            raise ValueError(
                f"No valid data files (PDF, TXT, MD) found in {PDF_DIR}."
            )

        logger.info("Total documents loaded: %d. Starting text splitting...", len(documents))

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
        logger.info("Successfully split into %d chunks (context size: %d, overlap: %d).", 
                    len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
        return chunks

    except Exception as e:
        logger.exception("Error during data ingestion/splitting: %s", str(e))
        raise


def create_vectorstore():
    """Build a FAISS index from documents and save to disk."""
    try:
        logger.info("Starting vector store creation...")
        chunks = load_and_split_data()

        logger.info("Initializing embedding model: %s", EMBEDDING_MODEL)
        embeddings = _get_embeddings()
        
        logger.info("Generating embeddings and building FAISS index (this may take a while)...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        logger.info("Saving index to disk: %s", INDEX_PATH)
        vectorstore.save_local(INDEX_PATH)

        logger.info("Vector store successfully created and saved at %s", INDEX_PATH)
        return vectorstore
    except Exception as e:
        logger.error("Failed to create vector store: %s", str(e))
        raise


def load_vectorstore():
    """Load an existing FAISS index from disk."""
    try:
        logger.info("Loading vector store from: %s", INDEX_PATH)
        embeddings = _get_embeddings()
        return FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        logger.error("Failed to load vector store from disk: %s", str(e))
        raise


def get_vectorstore():
    """Return the vectorstore — create it if it doesn't exist yet."""
    index_file = os.path.join(INDEX_PATH, "index.faiss")
    if os.path.exists(index_file):
        return load_vectorstore()
    else:
        logger.info("No index found at %s. Initializing first-time setup.", INDEX_PATH)
        return create_vectorstore()
