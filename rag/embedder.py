"""HuggingFace sentence-transformer embeddings via LangChain's embeddings
abstraction. Going through langchain_community.embeddings.HuggingFaceEmbeddings
instead of calling sentence-transformers directly means store.py and
retriever.py only ever depend on LangChain's Embeddings interface, so the
model backing them can be swapped without touching either.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # free, runs locally, no API key needed


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the embeddings object used to embed and query failure documents."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
