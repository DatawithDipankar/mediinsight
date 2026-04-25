"""
chroma_store.py
---------------
Manages ChromaDB collections for PubMed article storage and retrieval.
Uses a TF-IDF embedding function (sklearn) — no external model downloads required.

Provides:
  - get_client()               → PersistentClient
  - get_or_create_collection() → Collection
  - ingest_articles()          → int  (count ingested)
  - retrieve_documents()       → list[dict]
  - get_collection_stats()     → dict
"""
# Suppress protobuf conflict on Streamlit Cloud
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from __future__ import annotations

import json
from typing import Any

import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction, Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Constants ────────────────────────────────────────────────────────────────

COLLECTION_NAME = "pubmed_intermittent_fasting"
DB_PATH         = "./chroma_db"


# ── TF-IDF Embedding Function ────────────────────────────────────────────────

class TFIDFEmbeddingFunction(EmbeddingFunction):
    """
    Wraps sklearn's TfidfVectorizer as a ChromaDB EmbeddingFunction.

    The vectorizer must be fitted on the full corpus BEFORE any documents are
    added to or queried from the collection — this ensures query and document
    vectors share the same vocabulary and dimensionality.

    Usage:
        ef = TFIDFEmbeddingFunction(max_features=2048)
        ef.fit(all_document_texts)          # fit once
        collection = client.get_or_create_collection(..., embedding_function=ef)
        collection.upsert(...)              # uses fitted vectorizer
        collection.query(...)              # same vectorizer, same dim ✓
    """

    def __init__(self, max_features: int = 2048):
        self.vectorizer  = TfidfVectorizer(
            max_features  = max_features,
            sublinear_tf  = True,           # log(1 + tf) — dampens high-freq terms
            ngram_range   = (1, 2),         # unigrams + bigrams for better recall
            min_df        = 1,
            strip_accents = "unicode",
        )
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        """Fit the vectorizer on the entire document corpus."""
        self.vectorizer.fit(corpus)
        self._fitted = True
        vocab_size = len(self.vectorizer.vocabulary_)
        print(f"  [TF-IDF] Fitted on {len(corpus)} docs | vocab size: {vocab_size:,}")

    def __call__(self, input: list[str]) -> Embeddings:
        if not self._fitted:
            raise RuntimeError(
                "TFIDFEmbeddingFunction must be fitted before use. "
                "Call ef.fit(corpus) with all documents first."
            )
        matrix = self.vectorizer.transform(input).toarray().astype(float)
        return matrix.tolist()


# ── Client & collection helpers ──────────────────────────────────────────────

def get_client() -> chromadb.PersistentClient:
    """Return (or create) a persistent ChromaDB client."""
    return chromadb.PersistentClient(
        path     = DB_PATH,
        settings = Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(
    client         : chromadb.PersistentClient,
    embedding_func : EmbeddingFunction | None = None,
    name           : str = COLLECTION_NAME,
) -> chromadb.Collection:
    """
    Get an existing collection or create it if it doesn't exist.
    Pass a fitted TFIDFEmbeddingFunction as `embedding_func`.
    """
    kwargs: dict[str, Any] = {
        "name"     : name,
        "metadata" : {"hnsw:space": "cosine"},
    }
    if embedding_func is not None:
        kwargs["embedding_function"] = embedding_func

    return client.get_or_create_collection(**kwargs)


# ── Document preparation ─────────────────────────────────────────────────────

def _flatten_abstract(abstract: dict | str) -> str:
    """Convert structured abstract dict into a readable plain-text string."""
    if isinstance(abstract, str):
        return abstract
    return " ".join(
        f"[{label}] {text}" for label, text in abstract.items() if text
    )


def build_document(article: dict[str, Any]) -> str:
    """
    Combine all article fields into a single text document for embedding.
    The richer the text, the better the semantic search.
    """
    abstract_text = _flatten_abstract(article.get("abstract", {}))
    return (
        f"Title: {article.get('title', 'No Title')}\n"
        f"Authors: {article.get('authors', 'Unknown')}\n"
        f"Journal: {article.get('journal', 'Unknown Journal')}\n"
        f"Year: {article.get('publication_date', 'Unknown')}\n"
        f"Abstract: {abstract_text}"
    )


def _build_metadata(article: dict[str, Any]) -> dict[str, str]:
    """Flatten article into ChromaDB-compatible metadata (str values only)."""
    abstract = article.get("abstract", {})
    abstract_str = json.dumps(abstract) if isinstance(abstract, dict) else str(abstract)
    return {
        "pmid"            : str(article.get("pmid", "")),
        "title"           : str(article.get("title", "No Title")),
        "authors"         : str(article.get("authors", "Unknown")),
        "journal"         : str(article.get("journal", "Unknown Journal")),
        "publication_date": str(article.get("publication_date", "Unknown")),
        "abstract"        : abstract_str,
    }


# ── Ingestion ────────────────────────────────────────────────────────────────

def ingest_articles(
    articles   : list[dict[str, Any]],
    collection : chromadb.Collection,
    batch_size : int = 100,
) -> int:
    """
    Upsert PubMed article dicts into a ChromaDB collection in batches.

    Uses upsert so re-running is safe — duplicates are overwritten.
    Returns the number of articles successfully ingested.
    """
    if not articles:
        print("[ingest] No articles to ingest.")
        return 0

    ingested = 0
    for i in range(0, len(articles), batch_size):
        batch      = articles[i : i + batch_size]
        ids        = [str(a["pmid"])      for a in batch]
        documents  = [build_document(a)   for a in batch]
        metadatas  = [_build_metadata(a)  for a in batch]

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        ingested += len(batch)
        print(f"  [ingest] Upserted {ingested:>4} / {len(articles)} articles")

    return ingested


# ── Retrieval ────────────────────────────────────────────────────────────────

def retrieve_documents(
    query      : str,
    collection : chromadb.Collection,
    n_results  : int = 5,
) -> list[dict[str, Any]]:
    """
    Semantic search — returns top-n articles ranked by cosine similarity.

    Each result dict contains:
        pmid, title, authors, journal, publication_date,
        abstract (dict or str), distance (float), document (str)
    """
    results = collection.query(query_texts=[query], n_results=n_results)

    output = []
    for idx in range(len(results["ids"][0])):
        meta = results["metadatas"][0][idx]
        try:
            abstract = json.loads(meta.get("abstract", "{}"))
        except (json.JSONDecodeError, TypeError):
            abstract = meta.get("abstract", "")

        output.append({
            "pmid"            : meta.get("pmid"),
            "title"           : meta.get("title"),
            "authors"         : meta.get("authors"),
            "journal"         : meta.get("journal"),
            "publication_date": meta.get("publication_date"),
            "abstract"        : abstract,
            "distance"        : results["distances"][0][idx],
            "document"        : results["documents"][0][idx],
        })

    return output


# ── Stats ─────────────────────────────────────────────────────────────────────

def get_collection_stats(collection: chromadb.Collection) -> dict[str, Any]:
    """Return basic statistics about a collection."""
    return {
        "collection_name": collection.name,
        "total_documents": collection.count(),
        "metadata"       : collection.metadata,
    }
