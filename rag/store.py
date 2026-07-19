"""Chroma vector store via LangChain's langchain_chroma integration. Using
LangChain's Chroma wrapper instead of the chromadb client directly means
ingestion.py's Documents and embedder.py's Embeddings plug straight in with
no Chroma-specific glue, and the store itself could be swapped for another
LangChain vector store (FAISS, Pinecone, ...) without changing callers.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

COLLECTION_NAME = "validation_failures"
PERSIST_DIR = Path(__file__).parent.parent / "chroma_store"  # persisted at project root


def build_store(documents: list[Document], embeddings: Embeddings) -> Chroma:
    """Create (or overwrite) the persistent store, embedding the given documents."""
    return Chroma.from_documents( # embeds each document from list as vector and saves to disk
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
    )


def get_store(embeddings: Embeddings) -> Chroma:
    """Open the persistent store, creating an empty collection on first use."""
    return Chroma( # opens a store that already exists
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )


def add_documents(store: Chroma, documents: list[Document]) -> None:
    """Embed and add new documents to an already-open store."""
    if documents:
        store.add_documents(documents)
