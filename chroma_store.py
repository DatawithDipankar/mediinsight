"""
Vector store replacement — no ChromaDB dependency.
Uses TF-IDF + cosine similarity via scikit-learn.
Drop-in replacement: same function signatures as the original.
"""

import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── In-memory store (persists for the session) ────────────────────────────────
_store = {
    "documents": [],   # list of text strings
    "metadatas": [],   # list of metadata dicts
    "ids": [],         # list of string IDs
    "matrix": None,    # TF-IDF matrix (numpy array)
}
_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=1024,
    stop_words="english"
)
_fitted = False


# ── Public API (matches original chroma_store.py signatures) ──────────────────

class TFIDFEmbeddingFunction:
    """Kept for import compatibility — not used internally anymore."""
    def __call__(self, texts):
        return _vectorizer.transform(texts).toarray().tolist()


def get_client():
    """No-op — kept for import compatibility."""
    return None


def get_or_create_collection(client=None, name="pubmed"):
    """No-op — kept for import compatibility."""
    return None


def build_document(article: dict) -> str:
    """Combine article fields into a single searchable text block."""
    parts = []
    if article.get("title"):
        parts.append(f"Title: {article['title']}")
    if article.get("authors"):
        parts.append(f"Authors: {article['authors']}")
    if article.get("journal"):
        parts.append(f"Journal: {article['journal']}")
    if article.get("year"):
        parts.append(f"Year: {article['year']}")
    if article.get("abstract"):
        parts.append(f"Abstract: {article['abstract']}")
    return "\n".join(parts)


def ingest_articles(articles: list, collection=None):
    """
    Fit TF-IDF on full corpus and store all articles in memory.
    Safe to re-run — clears previous data first.
    """
    global _fitted, _store

    if not articles:
        return

    # Reset store
    _store = {"documents": [], "metadatas": [], "ids": [], "matrix": None}

    documents = [build_document(a) for a in articles]
    metadatas = [{
        "pmid":     str(a.get("pmid", "")),
        "title":    a.get("title", ""),
        "authors":  a.get("authors", ""),
        "journal":  a.get("journal", ""),
        "year":     str(a.get("year", "")),
        "abstract": a.get("abstract", ""),
    } for a in articles]
    ids = [str(a.get("pmid", i)) for i, a in enumerate(articles)]

    # Fit vectoriser on full corpus
    matrix = _vectorizer.fit_transform(documents)
    _fitted = True

    _store["documents"] = documents
    _store["metadatas"] = metadatas
    _store["ids"]       = ids
    _store["matrix"]    = matrix

    return len(articles)


def retrieve_documents(query: str, n_results: int = 5, collection=None) -> list:
    """
    Semantic search via cosine similarity.
    Returns list of metadata dicts for the top-n results.
    """
    global _fitted

    if not _fitted or _store["matrix"] is None:
        return []

    query_vec = _vectorizer.transform([query])
    scores    = cosine_similarity(query_vec, _store["matrix"])[0]
    top_idx   = scores.argsort()[::-1][:n_results]

    results = []
    for i in top_idx:
        meta = _store["metadatas"][i].copy()
        meta["score"] = float(scores[i])
        results.append(meta)

    return results


def get_collection_stats(collection=None) -> dict:
    """Returns current store size."""
    return {
        "count": len(_store["documents"]),
        "fitted": _fitted
    }
