# 🩺 MediAssist AI

A clinical decision-support tool that helps healthcare providers quickly access evidence-based insights on **intermittent fasting (IF)** as a treatment for obesity, Type 2 Diabetes, and metabolic disorders.

Built for **MediInsight Health Solutions** as part of a virtual internship project.

---

## Overview

MediAssist AI combines a **Retrieval-Augmented Generation (RAG)** pipeline with a **Streamlit UI** to let clinicians and research assistants:

- Search PubMed for relevant research articles
- Ingest articles into a ChromaDB vector store
- Ask natural-language questions and receive evidence-based answers powered by **Llama 3** via Groq

---

## Project Structure

```
mediinsight/
├── app.py              # Streamlit UI - main entry point
├── rag.py              # RAG pipeline (retrieve → prompt → generate)
├── chroma_store.py     # ChromaDB vector store management
├── pubmed.py           # PubMed API retrieval (search + fetch)
├── pipeline.py         # End-to-end ingestion pipeline (CLI)
├── mock_data.py        # Synthetic article generator for offline testing
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed to Git)
└── chroma_db/          # Persistent vector DB (auto-created, not committed)
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/mediinsight.git
cd mediinsight
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

---

## Running the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## How to Use

1. **Search** - Enter a search term in the sidebar (e.g. `intermittent fasting obesity`) and click **Search PubMed**
2. **Ingest** - Click **Ingest into Vector Store** to embed and store the retrieved articles in ChromaDB
3. **Ask** - Type a clinical question in the main query bar and receive an evidence-based answer with PMID citations

---

## Architecture

```
PubMed API
    │
    ▼
pubmed.py          ← searches & fetches articles (PMIDs + abstracts)
    │
    ▼
chroma_store.py    ← TF-IDF embeddings + ChromaDB upsert / retrieval
    │
    ▼
rag.py             ← builds context prompt → calls Llama 3 via Groq
    │
    ▼
app.py             ← Streamlit UI (sidebar search + chat interface)
```

### Key components

| Component             | Description                                                          |
|-----------------------|----------------------------------------------------------------------|
| **ChromaDB**          | Persistent vector store using cosine similarity for semantic search  |
| **TF-IDF Embeddings** | Local sklearn-based embeddings, no external model downloads required |
| **Groq / Llama 3**    | `llama-3.3-70b-versatile` for fast, high-quality response generation |
| **PubMed eutils API** | NCBI's free API for searching and fetching biomedical literature     |

---

## Running the CLI Pipeline (optional)

To ingest articles from the terminal without the UI:

```bash
python pipeline.py
```

This will search PubMed, fetch up to 300 articles, ingest them into ChromaDB, and run demo semantic queries.

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key for Llama 3 access |

---

## .gitignore

Make sure your `.gitignore` includes:

```
.env
chroma_db/
.venv/
__pycache__/
*.pyc
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | 1.56.0 | Web UI framework |
| `chromadb` | 1.5.8 | Vector store |
| `groq` | 1.2.0 | Llama 3 API client |
| `scikit-learn` | 1.8.0 | TF-IDF embeddings |
| `requests` | 2.33.1 | PubMed API calls |
| `python-dotenv` | 1.2.2 | Environment variable loading |

---

## Acknowledgements

- [PubMed / NCBI eutils](https://www.ncbi.nlm.nih.gov/home/develop/api/) - biomedical literature API
- [ChromaDB](https://www.trychroma.com/) - open-source vector database
- [Groq](https://groq.com/) - fast LLM inference
- [Streamlit](https://streamlit.io/) - Python web app framework
