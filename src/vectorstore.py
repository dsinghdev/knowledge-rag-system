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


from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader

# ... (rest of imports)

def load_and_split_data():
    """Load data from various formats in the data/ folder and subfolders."""
    print(f"📂 Scanning directory: {PDF_DIR}")
    
    # Load PDFs
    pdf_loader = DirectoryLoader(PDF_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    # Load Text and Markdown
    text_loader = DirectoryLoader(PDF_DIR, glob="**/*.txt", loader_cls=TextLoader)
    md_loader = DirectoryLoader(PDF_DIR, glob="**/*.md", loader_cls=TextLoader)

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
            f"No valid data files found in data folder. "
            "Please add some investment documents (PDF, TXT, or MD)."
        )

    print(f"  Loaded {len(documents)} document pages/files total.")

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
    print(f"  Split into {len(chunks)} chunks")
    return chunks


def create_vectorstore():
    """Build a FAISS index from documents and save to disk."""
    print("📄 Loading and splitting data...")
    chunks = load_and_split_data()

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
