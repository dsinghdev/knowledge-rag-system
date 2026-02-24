"""
Vector store management: PDF loading, chunking, FAISS index creation & loading.
"""

import os
import glob

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    PDF_DIR,
    INDEX_PATH,
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def _get_embeddings():
    """Return the embedding model instance."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


def load_and_split_pdfs():
    """Load every PDF in data/ and split into chunks."""
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {PDF_DIR}. "
            "Please add at least one Income Tax PDF."
        )

    documents = []
    for pdf_path in pdf_files:
        print(f"  Loading: {os.path.basename(pdf_path)}")
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

    print(f"  Loaded {len(documents)} pages from {len(pdf_files)} PDF(s)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"  Split into {len(chunks)} chunks")
    return chunks


def create_vectorstore():
    """Build a FAISS index from PDFs and save to disk."""
    print("📄 Loading and splitting PDFs...")
    chunks = load_and_split_pdfs()

    print("🔢 Creating embeddings & FAISS index...")
    embeddings = _get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)

    print("✅ Vector store saved to disk.")
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
    if os.path.exists(INDEX_PATH):
        print("📂 Loading existing vector store...")
        return load_vectorstore()
    else:
        return create_vectorstore()
