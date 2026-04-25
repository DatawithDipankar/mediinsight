"""
pipeline.py
-----------
End-to-end MediAssist AI pipeline:

  Step 1 — Search PubMed for intermittent-fasting article IDs (PMIDs).
  Step 2 — Fetch full abstracts & metadata for each PMID.
  Step 3 — Fit a TF-IDF embedding model on the full corpus.
  Step 4 — Ingest all articles into a ChromaDB persistent vector store.
  Step 5 — Run demo semantic queries to verify retrieval.

Network note
------------
The sandbox blocks outbound calls to eutils.ncbi.nlm.nih.gov.
The pipeline detects this and falls back to realistic synthetic data
so every other component can be exercised end-to-end.
On a machine with unrestricted internet the real PubMedRetriever
calls will run instead.
"""

import requests

from chroma_store import (
    TFIDFEmbeddingFunction,
    get_client,
    get_or_create_collection,
    build_document,
    ingest_articles,
    retrieve_documents,
    get_collection_stats,
)

# ── Configuration ──────────────────────────────────────────────────────────────

SEARCH_TERM  = "intermittent fasting obesity type 2 diabetes metabolic disorders"
MAX_ARTICLES = 300
DEMO_QUERIES = [
    "What are the metabolic effects of intermittent fasting on blood glucose?",
    "How does the 16:8 fasting protocol affect weight loss in obese patients?",
    "Intermittent fasting and insulin resistance outcomes",
    "Alternate day fasting compared to continuous caloric restriction",
]


# ── Network probe ──────────────────────────────────────────────────────────────

def _pubmed_reachable() -> bool:
    """Return True only when PubMed eutils responds with valid XML."""
    try:
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": "test", "retmax": "1", "retmode": "xml"},
            timeout=5,
        )
        return r.status_code == 200 and b"<eSearchResult>" in r.content
    except Exception:
        return False


# ── Step 1 – Retrieve PMIDs ────────────────────────────────────────────────────

def step1_retrieve_pmids(max_results: int = MAX_ARTICLES) -> tuple[list[str], str]:
    print("\n" + "=" * 60)
    print("STEP 1 — Searching PubMed for article IDs (PMIDs)")
    print("=" * 60)
    print(f"  Search term : {SEARCH_TERM}")
    print(f"  Max results : {max_results}")

    if _pubmed_reachable():
        from pubmed import PubMedRetriever
        pmids  = PubMedRetriever.search_pubmed_articles(SEARCH_TERM, max_results)
        source = "PubMed API"
    else:
        print("\n  ⚠  PubMed eutils unreachable (network policy).")
        print("     Falling back to synthetic mock data for demo purposes.")
        print("     → Run on a machine with open internet for real PubMed data.\n")
        from mock_data import generate_mock_pmids
        pmids  = generate_mock_pmids(max_results)
        source = "mock data"

    print(f"  Source      : {source}")
    print(f"  Retrieved   : {len(pmids)} PMIDs")
    print(f"  Sample IDs  : {pmids[:5]}")
    return pmids, source


# ── Step 2 – Fetch article details ────────────────────────────────────────────

def step2_fetch_articles(pmids: list[str], source: str) -> list[dict]:
    print("\n" + "=" * 60)
    print("STEP 2 — Fetching article abstracts & metadata")
    print("=" * 60)

    if source == "PubMed API":
        from pubmed import PubMedRetriever
        print(f"  Fetching {len(pmids)} articles from PubMed in batches of 100 …")
        articles = PubMedRetriever.fetch_pubmed_abstracts(pmids)
    else:
        from mock_data import generate_mock_articles
        print(f"  Generating {len(pmids)} synthetic articles …")
        articles = generate_mock_articles(len(pmids))

    print(f"\n  Source      : {source}")
    print(f"  Total       : {len(articles)} articles fetched")

    if articles:
        s  = articles[0]
        ab = s["abstract"]
        sn = next(iter(ab.values()), "") if isinstance(ab, dict) else str(ab)
        print("\n  ── Sample article ──────────────────────────────────────")
        print(f"  PMID     : {s['pmid']}")
        print(f"  Title    : {s['title']}")
        print(f"  Journal  : {s['journal']}  ({s['publication_date']})")
        print(f"  Authors  : {s['authors'][:70]}")
        print(f"  Abstract : {sn[:150]} …")

    return articles


# ── Step 3 – Fit TF-IDF on full corpus ────────────────────────────────────────

def step3_fit_embeddings(articles: list[dict]) -> TFIDFEmbeddingFunction:
    print("\n" + "=" * 60)
    print("STEP 3 — Fitting TF-IDF embedding model on full corpus")
    print("=" * 60)
    print(f"  Documents   : {len(articles)}")
    print(f"  Strategy    : TF-IDF with bigrams, sublinear scaling, 2048 features")

    ef    = TFIDFEmbeddingFunction(max_features=1024)
    corpus = [build_document(a) for a in articles]
    ef.fit(corpus)

    print("  ✓ Embedding model ready")
    return ef


# ── Step 4 – Ingest into ChromaDB ─────────────────────────────────────────────

def step4_ingest(articles: list[dict], ef: TFIDFEmbeddingFunction):
    print("\n" + "=" * 60)
    print("STEP 4 — Creating ChromaDB collection & ingesting articles")
    print("=" * 60)

    client     = get_client()
    collection = get_or_create_collection(client, embedding_func=ef)

    print(f"  Collection  : {collection.name}")
    print(f"  DB path     : ./chroma_db  (persistent on disk)")
    print(f"  Embedding   : TF-IDF (local, no API key required)")
    print(f"  Upserting   : {len(articles)} articles in batches …\n")

    count = ingest_articles(articles, collection)
    stats = get_collection_stats(collection)

    print(f"\n  ✓ Ingestion complete")
    print(f"    Articles upserted  : {count}")
    print(f"    Total in DB        : {stats['total_documents']}")
    print(f"    Similarity metric  : {stats['metadata'].get('hnsw:space', 'cosine')}")

    return client, collection, ef


# ── Step 5 – Demo semantic retrieval ──────────────────────────────────────────

def step5_demo_retrieval(collection, ef: TFIDFEmbeddingFunction) -> None:
    print("\n" + "=" * 60)
    print("STEP 5 — Demo: Semantic retrieval from vector store")
    print("=" * 60)

    for query in DEMO_QUERIES:
        print(f"\n  ❓ Query: \"{query}\"")
        results = retrieve_documents(query, collection, n_results=3)

        for rank, r in enumerate(results, 1):
            ab  = r["abstract"]
            sn  = next(iter(ab.values()), "") if isinstance(ab, dict) else str(ab)
            print(f"\n    [{rank}] PMID {r['pmid']}  |  cosine dist = {r['distance']:.4f}")
            print(f"        Title   : {r['title'][:72]} …")
            print(f"        Journal : {r['journal']}  ({r['publication_date']})")
            print(f"        Snippet : {sn[:110]} …")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" 🩺  MediAssist AI — PubMed → ChromaDB Ingestion Pipeline")
    print("=" * 60)

    pmids, source          = step1_retrieve_pmids()
    articles               = step2_fetch_articles(pmids, source)
    ef                     = step3_fit_embeddings(articles)
    client, collection, ef = step4_ingest(articles, ef)
    step5_demo_retrieval(collection, ef)

    print("\n" + "=" * 60)
    print("  ✅  Pipeline complete — ChromaDB is ready for RAG queries.")
    print("=" * 60 + "\n")
