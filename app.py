"""
app.py  —  MediAssist AI · Streamlit UI
Run: streamlit run app.py
"""

import os, json, requests
from xml.etree import ElementTree
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from chroma_store import (
    TFIDFEmbeddingFunction, get_client, get_or_create_collection,
    build_document, ingest_articles, retrieve_documents, get_collection_stats
)
from rag import MediAssistRAG

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="MediAssist AI", page_icon="🩺",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
    --bg:#0d1117; --bg-card:#161b22; --bg-input:#1c2128;
    --border:#30363d; --teal:#39d0c8; --teal-dim:rgba(57,208,200,0.12);
    --teal-glow:rgba(57,208,200,0.35); --green:#2ea043; --amber:#d29922;
    --text:#e6edf3; --text2:#8b949e; --text3:#484f58;
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:var(--bg)!important;color:var(--text)!important;}
.medi-header{display:flex;align-items:center;gap:18px;padding:28px 0 20px;border-bottom:1px solid var(--border);margin-bottom:24px;}
.medi-logo{width:52px;height:52px;background:linear-gradient(135deg,#39d0c8,#2ea043);border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:26px;box-shadow:0 0 24px var(--teal-glow);flex-shrink:0;}
.medi-title{font-family:'DM Serif Display',serif;font-size:2rem;margin:0;line-height:1.1;}
.medi-subtitle{font-size:.78rem;color:var(--text2);margin:4px 0 0;letter-spacing:.05em;text-transform:uppercase;}
section[data-testid="stSidebar"]{background:var(--bg-card)!important;border-right:1px solid var(--border)!important;}
section[data-testid="stSidebar"] *{color:var(--text)!important;}
.sb-brand{font-family:'DM Serif Display',serif;font-size:1.25rem;color:var(--teal)!important;margin-bottom:2px;}
.sb-tag{font-size:.72rem;color:var(--text2)!important;margin-bottom:18px;}
.sb-label{font-size:.65rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--text3)!important;margin:18px 0 8px;}
.stat-row{display:flex;gap:8px;margin:14px 0;}
.stat-badge{flex:1;background:var(--bg-input);border:1px solid var(--border);border-radius:8px;padding:10px;text-align:center;}
.stat-badge .num{font-family:'JetBrains Mono',monospace;font-size:1.25rem;color:var(--teal);display:block;}
.stat-badge .lbl{font-size:.62rem;color:var(--text2);text-transform:uppercase;letter-spacing:.06em;}
.stTextInput input,.stNumberInput input{background:var(--bg-input)!important;border:1px solid var(--border)!important;border-radius:8px!important;color:var(--text)!important;}
.stTextInput input:focus{border-color:var(--teal)!important;box-shadow:0 0 0 3px var(--teal-dim)!important;}
.stButton>button{background:linear-gradient(135deg,#39d0c8,#2ea043)!important;color:#0d1117!important;font-weight:600!important;border:none!important;border-radius:8px!important;width:100%!important;transition:opacity .2s,transform .15s!important;}
.stButton>button:hover{opacity:.85!important;transform:translateY(-1px)!important;}
.art-card{background:var(--bg-input);border:1px solid var(--border);border-radius:8px;padding:11px 13px;margin-bottom:7px;transition:border-color .2s;}
.art-card:hover{border-color:var(--teal);}
.art-pmid{font-family:'JetBrains Mono',monospace;font-size:.68rem;color:var(--teal);margin-bottom:3px;}
.art-title{font-size:.78rem;font-weight:500;line-height:1.4;margin-bottom:3px;}
.art-meta{font-size:.68rem;color:var(--text2);}
.chat-bubble{display:flex;gap:14px;margin-bottom:22px;animation:fadeSlide .3s ease;}
@keyframes fadeSlide{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.chat-avatar{width:36px;height:36px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:17px;flex-shrink:0;margin-top:2px;}
.av-user{background:#21262d;border:1px solid var(--border);}
.av-ai{background:linear-gradient(135deg,#39d0c8,#2ea043);box-shadow:0 0 14px var(--teal-glow);}
.chat-label{font-size:.63rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--text3);margin-bottom:5px;}
.chat-text{background:var(--bg-card);border:1px solid var(--border);border-radius:10px;padding:15px 17px;font-size:.88rem;line-height:1.75;color:var(--text);}
.chat-text.usr{background:var(--bg-input);}
.src-row{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px;}
.src-pill{font-family:'JetBrains Mono',monospace;font-size:.65rem;background:var(--teal-dim);color:var(--teal);border:1px solid rgba(57,208,200,.25);border-radius:20px;padding:3px 10px;}
.empty-state{text-align:center;padding:55px 20px;color:var(--text3);}
.empty-icon{font-size:2.8rem;margin-bottom:14px;}
.empty-title{font-family:'DM Serif Display',serif;font-size:1.25rem;color:var(--text2);margin-bottom:8px;}
.empty-hint{font-size:.8rem;line-height:1.7;}
.alert{border-radius:8px;padding:11px 15px;margin:10px 0;font-size:.83rem;}
.a-ok{background:rgba(46,160,67,.12);border:1px solid rgba(46,160,67,.3);color:#3fb950;}
.a-info{background:rgba(57,208,200,.08);border:1px solid rgba(57,208,200,.25);color:var(--teal);}
.a-warn{background:rgba(210,153,34,.1);border:1px solid rgba(210,153,34,.3);color:var(--amber);}
.sug-btn button{background:var(--bg-input)!important;color:var(--text2)!important;border:1px solid var(--border)!important;font-size:.78rem!important;font-weight:400!important;text-align:left!important;}
.sug-btn button:hover{border-color:var(--teal)!important;color:var(--teal)!important;}
hr{border-color:var(--border)!important;}
#MainMenu,footer{visibility:hidden;}
header{visibility:visible !important;}
.block-container{padding-top:2rem!important;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k,v in {"chat_history":[],"collection":None,"ef":None,"rag":None,
             "ingested_count":0,"search_results":[],"last_search":""}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── PubMed helpers ────────────────────────────────────────────────────────────
SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
FETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

def pubmed_search(term, max_results=50):
    try:
        r = requests.get(SEARCH_URL, params={"db":"pubmed","term":term,"retmax":max_results,"retmode":"xml"}, timeout=10)
        root = ElementTree.fromstring(r.content)
        return [el.text for el in root.findall(".//Id")]
    except Exception as e:
        st.error(f"PubMed search error: {e}"); return []

def pubmed_fetch(pmids):
    articles = []
    try:
        r = requests.get(FETCH_URL, params={"db":"pubmed","id":",".join(pmids),"retmode":"xml"}, timeout=20)
        root = ElementTree.fromstring(r.content)
        for art in root.findall(".//PubmedArticle"):
            pmid  = art.find(".//PMID").text
            title = art.find(".//ArticleTitle")
            title = title.text if title is not None else "No Title"
            secs  = art.findall(".//AbstractText")
            abstract = {s.attrib.get("Label","SUMMARY"):s.text for s in secs if s.text} if secs else {"SUMMARY":"No Abstract"}
            journal = art.find(".//Journal/Title")
            journal = journal.text if journal is not None else "Unknown"
            year    = art.find(".//PubDate/Year")
            year    = year.text if year is not None else "Unknown"
            authors = [f"{a.find('.//ForeName').text} {a.find('.//LastName').text}"
                       for a in art.findall(".//Author")
                       if a.find(".//ForeName") is not None and a.find(".//LastName") is not None]
            articles.append({"pmid":pmid,"title":title,"abstract":abstract,
                              "journal":journal,"publication_date":year,
                              "authors":", ".join(authors) if authors else "No Authors"})
    except Exception as e:
        st.error(f"PubMed fetch error: {e}")
    return articles

def build_rag(articles):
    ef = TFIDFEmbeddingFunction()
    client = get_client()
    collection = get_or_create_collection(client)
    ingest_articles(articles, collection)
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY", "")
    rag = MediAssistRAG(collection, groq_api_key=groq_api_key)
    return ef, collection, rag

# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sb-brand">🩺 MediAssist AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-tag">Evidence-based clinical insights</div>', unsafe_allow_html=True)

    ic = st.session_state.ingested_count
    sr = len(st.session_state.search_results)
    ch = len(st.session_state.chat_history) // 2
    st.markdown(f"""<div class="stat-row">
      <div class="stat-badge"><span class="num">{ic}</span><span class="lbl">Ingested</span></div>
      <div class="stat-badge"><span class="num">{sr}</span><span class="lbl">Found</span></div>
      <div class="stat-badge"><span class="num">{ch}</span><span class="lbl">Queries</span></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sb-label">PubMed Search</div>', unsafe_allow_html=True)

    search_term = st.text_input("Search term", value="intermittent fasting type 2 diabetes",
                                 placeholder="e.g. intermittent fasting obesity",
                                 label_visibility="collapsed")
    max_res = st.slider("Max articles", 10, 300, 50, step=10)

    if st.button("Search PubMed"):
        if search_term.strip():
            with st.spinner("Searching PubMed…"):
                pmids = pubmed_search(search_term.strip(), max_res)
            if pmids:
                with st.spinner(f"Fetching {len(pmids)} articles…"):
                    arts = pubmed_fetch(pmids)
                st.session_state.search_results = arts
                st.session_state.last_search    = search_term.strip()
                st.markdown(f'<div class="alert a-info">✓ Found <b>{len(arts)}</b> articles</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert a-warn">⚠ No results found.</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter a search term.")

    if st.session_state.search_results:
        st.markdown(f'<div class="sb-label">{len(st.session_state.search_results)} Articles Found</div>', unsafe_allow_html=True)
        for art in st.session_state.search_results[:8]:
            ab = art["abstract"]
            st.markdown(f"""<div class="art-card">
              <div class="art-pmid">PMID {art['pmid']}</div>
              <div class="art-title">{art['title'][:82]}{'…' if len(art['title'])>82 else ''}</div>
              <div class="art-meta">{art['journal']} · {art['publication_date']}</div>
            </div>""", unsafe_allow_html=True)
        if len(st.session_state.search_results) > 8:
            st.markdown(f'<div style="text-align:center;font-size:.7rem;color:var(--text3);padding:6px">+ {len(st.session_state.search_results)-8} more articles</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="sb-label">⚡ Vector Store</div>', unsafe_allow_html=True)
        if st.button("Ingest into Vector Store"):
            arts = st.session_state.search_results
            with st.spinner(f"Ingesting {len(arts)} articles…"):
                ef, collection, rag = build_rag(arts)
                st.session_state.ef = ef
                st.session_state.collection = collection
                st.session_state.rag = rag
                st.session_state.ingested_count = get_collection_stats()["count"]
            st.markdown(f'<div class="alert a-ok">✓ Ingested <b>{st.session_state.ingested_count}</b> articles. Ready!</div>', unsafe_allow_html=True)
            st.rerun()

    if st.session_state.ingested_count > 0:
        st.markdown("---")
        if st.button("Clear & Reset"):
            import shutil
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            shutil.rmtree("./chroma_db", ignore_errors=True)
            st.rerun()

# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
st.markdown("""<div class="medi-header">
  <div class="medi-logo">🩺</div>
  <div>
    <div class="medi-title">MediAssist AI</div>
    <div class="medi-subtitle">Intermittent Fasting · Clinical Research Assistant</div>
  </div>
</div>""", unsafe_allow_html=True)

if st.session_state.ingested_count > 0:
    st.markdown(f'<div class="alert a-ok">✅ &nbsp;<b>{st.session_state.ingested_count} articles</b> loaded from "<b>{st.session_state.last_search}</b>" — ask anything below.</div>', unsafe_allow_html=True)
elif st.session_state.search_results:
    st.markdown('<div class="alert a-warn">⚠ &nbsp;Articles found but not ingested. Click <b>"Ingest into Vector Store"</b> in the sidebar.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="alert a-info">💡 &nbsp;Search for PubMed articles in the sidebar → ingest them → then ask clinical questions here.</div>', unsafe_allow_html=True)

# Chat history
if not st.session_state.chat_history:
    st.markdown("""<div class="empty-state">
      <div class="empty-title">No queries yet</div>
      <div class="empty-hint">Search PubMed → Ingest articles → Ask clinical questions<br>about intermittent fasting, metabolic outcomes, and more.</div>
    </div>""", unsafe_allow_html=True)
else:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""<div class="chat-bubble">
              <div class="chat-avatar av-user">👤</div>
              <div style="flex:1"><div class="chat-label">Clinician</div>
              <div class="chat-text usr">{msg['content']}</div></div>
            </div>""", unsafe_allow_html=True)
        else:
            pills = "".join(f'<span class="src-pill">PMID {s}</span>' for s in msg.get("sources",[]))
            src   = f'<div class="src-row">{pills}</div>' if pills else ""
            st.markdown(f"""<div class="chat-bubble">
              <div class="chat-avatar av-ai">🩺</div>
              <div style="flex:1"><div class="chat-label">MediAssist AI · Llama 3</div>
              <div class="chat-text">{msg['content'].replace(chr(10),'<br>')}</div>
              {src}</div>
            </div>""", unsafe_allow_html=True)

# Query bar
st.markdown("---")
col1, col2 = st.columns([6,1])
with col1:
    query = st.text_input("Ask", placeholder="e.g. What are the effects of 16:8 fasting on HbA1c in Type 2 diabetes?",
                           label_visibility="collapsed", key="query_input")
with col2:
    send = st.button("Ask →", use_container_width=True)

# Suggestions
if not st.session_state.chat_history:
    st.markdown('<div style="margin-top:10px;font-size:.68rem;color:var(--text3);text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">Suggested questions</div>', unsafe_allow_html=True)
    sugs = [
        "What are the metabolic effects of 16:8 fasting on blood glucose?",
        "How does alternate-day fasting compare to caloric restriction?",
        "Which IF protocol is most effective for insulin resistance?",
        "What safety concerns exist for IF in diabetic patients?",
    ]
    c1, c2 = st.columns(2)
    for i, sug in enumerate(sugs):
        with (c1 if i%2==0 else c2):
            with st.container():
                st.markdown('<div class="sug-btn">', unsafe_allow_html=True)
                if st.button(sug, key=f"sug_{i}", use_container_width=True):
                    query = sug; send = True
                st.markdown('</div>', unsafe_allow_html=True)

# Process
if send and query and query.strip():
    if st.session_state.rag is None:
        st.markdown('<div class="alert a-warn">⚠ Please ingest articles first using the sidebar.</div>', unsafe_allow_html=True)
    else:
        st.session_state.chat_history.append({"role":"user","content":query.strip()})
        with st.spinner("Retrieving relevant studies and generating answer…"):
            rag      = st.session_state.rag
            articles = rag.retrieve(query.strip())
            prompt   = rag.build_prompt(query.strip(), articles)
            answer   = rag.generate(prompt)
            sources  = [a["pmid"] for a in articles]
        st.session_state.chat_history.append({"role":"assistant","content":answer,"sources":sources})
        st.rerun()
