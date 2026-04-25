"""
rag.py
------
Retrieval-Augmented Generation (RAG) pipeline for MediAssist AI.

Flow:
  1. User submits a natural-language question.
  2. ChromaDB is queried → top-k relevant PubMed articles retrieved.
  3. Retrieved articles are formatted into a context block.
  4. A structured prompt (context + query) is sent to Llama 3 via Groq.
  5. An evidence-based answer is returned to the caller.

Usage (standalone):
    python3 rag.py

Usage (as a module):
    from rag import MediAssistRAG
    rag = MediAssistRAG(collection, embedding_func)
    answer = rag.answer("Does 16:8 fasting help with Type 2 diabetes?")
    print(answer)
"""

from __future__ import annotations

import os
import textwrap
from typing import Any
from dotenv import load_dotenv
from groq import Groq
from chroma_store import retrieve_documents

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

LLAMA_MODEL   = "llama-3.3-70b-versatile"   # Llama 3 70B via Groq
DEFAULT_TOP_K = 5                   # articles to retrieve per query

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are MediAssist AI, a clinical decision-support assistant \
specialising in intermittent fasting (IF) as a treatment for obesity, Type 2 \
Diabetes, and metabolic disorders.

Your role is to help clinicians and research assistants quickly access \
evidence-based information. You answer questions using ONLY the PubMed research \
articles provided in the context below. 

Guidelines:
- Cite specific studies by their PMID when supporting a claim.
- Clearly distinguish between findings from different IF protocols \
  (e.g. 16:8, 5:2, alternate-day fasting).
- If the context does not contain enough information to answer confidently, \
  say so — do NOT fabricate citations or data.
- Keep answers concise, structured, and clinician-friendly.
- Where relevant, note study limitations or conflicting findings.
"""

# ── RAG Pipeline ──────────────────────────────────────────────────────────────

class MediAssistRAG:
    """
    RAG pipeline: ChromaDB retrieval → prompt construction → Groq/Llama generation.
    """

    def __init__(
        self,
        collection,
        groq_api_key : str | None = None,
        top_k        : int = DEFAULT_TOP_K,
        model        : str = LLAMA_MODEL,
    ):
        """
        Parameters
        ----------
        collection   : chromadb.Collection  — fitted, populated ChromaDB collection.
        groq_api_key : str | None           — Groq API key (falls back to
                                              GROQ_API_KEY env var if None).
        top_k        : int                  — number of articles to retrieve.
        model        : str                  — Groq model identifier.
        """
        self.collection = collection
        self.top_k      = top_k
        self.model      = model
        self.client     = Groq(api_key=groq_api_key or os.environ["GROQ_API_KEY"])

    # ── Step 1: Query the vector store ────────────────────────────────────────

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """
        Query ChromaDB with the user's natural-language question.
        Returns the top-k most semantically similar articles.
        """
        results = retrieve_documents(query, self.collection, n_results=self.top_k)
        return results

    # ── Step 2: Build context + prompt ────────────────────────────────────────

    @staticmethod
    def _format_article(article: dict[str, Any], rank: int) -> str:
        """Format a single retrieved article into a readable context block."""
        abstract = article.get("abstract", {})
        if isinstance(abstract, dict):
            abstract_text = " ".join(
                f"[{label}] {text}"
                for label, text in abstract.items()
                if text
            )
        else:
            abstract_text = str(abstract)

        return (
            f"--- Article {rank} ---\n"
            f"PMID       : {article.get('pmid', 'N/A')}\n"
            f"Title      : {article.get('title', 'N/A')}\n"
            f"Authors    : {article.get('authors', 'N/A')}\n"
            f"Journal    : {article.get('journal', 'N/A')} "
            f"({article.get('publication_date', 'N/A')})\n"
            f"Relevance  : cosine distance = {article.get('distance', 0):.4f}\n"
            f"Abstract   : {abstract_text}\n"
        )

    def build_prompt(self, query: str, articles: list[dict[str, Any]]) -> str:
        """
        Construct the user-turn prompt containing:
          - Context  : formatted retrieved articles
          - Query    : the clinician's question
        """
        if articles:
            context_blocks = "\n".join(
                self._format_article(a, i + 1) for i, a in enumerate(articles)
            )
        else:
            context_blocks = "No relevant articles were found in the vector store."

        prompt = textwrap.dedent(f"""
            CONTEXT — Retrieved PubMed Articles
            ====================================
            {context_blocks}

            QUERY
            =====
            {query}

            Using ONLY the articles provided above, give a clear, evidence-based
            answer to the query. Cite PMIDs where relevant.
        """).strip()

        return prompt

    # ── Step 3: Generate response via Groq / Llama 3 ─────────────────────────

    def generate(self, prompt: str) -> str:
        """
        Send the constructed prompt to Llama 3 (via Groq) and return the answer.
        """
        response = self.client.chat.completions.create(
            model    = self.model,
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature = 0.2,    # low temperature → factual, reproducible answers
            max_tokens  = 1024,
        )
        return response.choices[0].message.content

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def answer(self, query: str, verbose: bool = False) -> str:
        """
        End-to-end RAG: retrieve → prompt → generate → return answer string.

        Parameters
        ----------
        query   : the clinician's natural-language question.
        verbose : if True, print retrieved articles and the prompt before answering.
        """
        # Step 1 — retrieve
        articles = self.retrieve(query)

        if verbose:
            print(f"\n[RAG] Retrieved {len(articles)} articles for query:")
            print(f"      \"{query}\"")
            for i, a in enumerate(articles, 1):
                print(f"  [{i}] PMID {a['pmid']} | dist={a['distance']:.4f} | {a['title'][:60]} …")

        # Step 2 — build prompt
        prompt = self.build_prompt(query, articles)

        if verbose:
            print("\n[RAG] Prompt sent to Llama 3:")
            print("-" * 60)
            print(prompt[:800], "…")
            print("-" * 60)

        # Step 3 — generate
        answer = self.generate(prompt)
        return answer


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from mock_data import generate_mock_articles
    from chroma_store import (
        TFIDFEmbeddingFunction,
        get_client,
        get_or_create_collection,
        build_document,
        ingest_articles,
    )

    # ── Load / rebuild the collection ────────────────────────────────────────
    print("\n🩺  MediAssist AI — RAG Demo")
    print("=" * 60)

    print("\n[setup] Loading ChromaDB collection …")
    articles = generate_mock_articles(300)
    ef       = TFIDFEmbeddingFunction(max_features=1024)
    ef.fit([build_document(a) for a in articles])

    client     = get_client()
    collection = get_or_create_collection(client, embedding_func=ef)

    # Ingest only if empty
    if collection.count() == 0:
        print("[setup] Collection empty — ingesting 300 articles …")
        ingest_articles(articles, collection)
    else:
        print(f"[setup] Collection ready — {collection.count()} articles in DB")

    # ── Check for API key ────────────────────────────────────────────────────
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("\n⚠  GROQ_API_KEY environment variable not set.")
        print("   Set it with:  set GROQ_API_KEY=your_key_here  (Windows)")
        print("   Then re-run:  python3 rag.py")
        sys.exit(1)

    # ── Initialise RAG pipeline ──────────────────────────────────────────────
    rag = MediAssistRAG(collection=collection, groq_api_key=api_key)

    # ── Demo queries ─────────────────────────────────────────────────────────
    demo_queries = [
        "What are the effects of 16:8 intermittent fasting on blood glucose in Type 2 diabetes?",
        "How does alternate-day fasting compare to continuous caloric restriction for weight loss?",
    ]

    for query in demo_queries:
        print(f"\n{'='*60}")
        print(f"❓ Query: {query}")
        print("=" * 60)
        response = rag.answer(query, verbose=True)
        print(f"\n💬 MediAssist AI Answer:\n")
        print(response)

    print("\n" + "=" * 60)
    print("✅  RAG demo complete.")
    print("=" * 60)
