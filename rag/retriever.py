"""LangChain retriever over the Chroma store.

Two related but distinct requirements are handled here:

- get_retriever() uses Maximum Marginal Relevance (MMR) search, so historical
  failures returned for a query aren't all near-duplicates of each other
  (e.g. the same metric failing the same way across many runs) -- it
  optimizes for *diverse, relevant* results, not purely top-N-by-score.
- has_relevant_matches() runs a similarity search with a score threshold,
  used by rag.pipeline to decide whether the MMR results are actually
  relevant enough to ground a diagnosis, or whether the pipeline should
  fall back to an ungrounded response. MMR and a hard score cutoff don't
  compose in LangChain's retriever API (they're different `search_type`s),
  so this is kept as a separate, explicit check rather than folded into
  get_retriever's own search_kwargs.
"""

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

DEFAULT_K = 5
FETCH_K = 20  # candidate pool MMR selects the diverse top-k from
SCORE_THRESHOLD = 0.5


def get_retriever(store: Chroma, k: int = DEFAULT_K, gpu_model: str | None = None) -> VectorStoreRetriever:
    """Return an MMR retriever over the store, optionally filtered to one GPU model."""
    search_kwargs = {"k": k, "fetch_k": FETCH_K} # dict of settings for Chroma
    if gpu_model is not None:
        search_kwargs["filter"] = {"gpu_model": gpu_model}

    return store.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
    # returns store as "retriever" object, can be called with .invoke() to get back list of matching Documents
    # mmr search type grabs larger pool of candidates, and picks 5 that are relevant AND different from each other (prevents same failure being returned 5 times)


def has_relevant_matches(store: Chroma, query: str, k: int = DEFAULT_K, score_threshold: float = SCORE_THRESHOLD) -> bool:
    """Similarity search with a score threshold: True if at least one stored failure
    is relevant enough to the query to ground a diagnosis on."""
    results = store.similarity_search_with_relevance_scores(query, k=k)
    return any(score >= score_threshold for _, score in results)
