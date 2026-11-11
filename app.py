import streamlit as st
from src.config import LLM_MODEL, EMBEDDING_MODEL
from src.llm import check_ollama
from src.rag_pipeline import ingest_document, ask_question
from src.vector_store import get_user_documents, delete_document

st.set_page_config(page_title="PERSONAL RAG", page_icon="◼", layout="wide")
USER_ID = "demo_user_001"  # Demo only; replace with authenticated user ID for deployment.

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--ink:#080808;--panel:#111;--line:#292929;--white:#F4F1EA;--muted:#A7A39B;--orange:#FF5A1F}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(ellipse at 82% 0%,rgba(255,90,31,.075),transparent 30%),var(--ink);color:var(--white)}
[data-testid="stHeader"]{background:rgba(8,8,8,.92);border-bottom:1px solid var(--line)}
.block-container{max-width:1440px;padding-top:2.4rem;padding-bottom:4rem}
h1,h2,h3,h4{font-family:'Manrope',sans-serif!important;color:var(--white)!important;letter-spacing:-.04em}
.brand-kicker,.section-label,.tiny-mono,.metric-label{font-family:'DM Mono',monospace;text-transform:uppercase;letter-spacing:.15em}
.brand-kicker{color:var(--orange);font-size:.72rem;margin-bottom:.8rem}
.hero-title{font-family:'Manrope',sans-serif;font-weight:800;font-size:clamp(2.6rem,5vw,4.5rem);line-height:.98;letter-spacing:-.075em;color:var(--white)}
.hero-title span{color:var(--orange)}
.hero-copy,.upload-note{color:var(--muted);line-height:1.7}
.hero-copy{font-size:1rem;max-width:680px;margin-top:1rem}
.section-label{color:var(--muted);font-size:.68rem;margin:1.4rem 0 .65rem}
.tiny-mono{color:var(--muted);font-size:.65rem;letter-spacing:.08em}
.orange-rule{height:2px;width:46px;background:var(--orange);border-radius:10px;margin:.8rem 0 1.1rem}
.metric-card{background:linear-gradient(145deg,#171717,#101010);border:1px solid var(--line);border-radius:16px;padding:1rem 1.15rem;min-height:105px}
.metric-label{color:var(--muted);font-size:.66rem}
.metric-value{font-family:'Manrope',sans-serif;color:var(--white);font-size:1.7rem;font-weight:800;margin:.35rem 0}
.metric-accent{color:var(--orange)}
.surface-card{background:#111;border:1px solid var(--line);border-radius:18px;padding:1.25rem}
[data-testid="stSidebar"]{background:#0D0D0D;border-right:1px solid var(--line)}
[data-testid="stSidebar"]>div:first-child{padding-top:1.5rem}
[data-testid="stFileUploader"]{background:#141414;border:1px dashed #45413E;border-radius:14px;padding:.55rem}
[data-testid="stFileUploader"] section{background:transparent}
[data-testid="stFileUploader"] button{background:rgba(255,90,31,.13)!important;color:var(--white)!important;border:1px solid var(--orange)!important;border-radius:9px!important}
.stButton>button,.stDownloadButton>button{background:var(--orange)!important;color:#090909!important;border:1px solid var(--orange)!important;border-radius:10px!important;font-weight:800!important;min-height:2.65rem;transition:all .18s ease}
.stButton>button:hover,.stDownloadButton>button:hover{background:#FF7848!important;border-color:#FF7848!important;color:#090909!important;transform:translateY(-1px)}
[data-testid="stChatMessage"]{background:#111;border:1px solid var(--line);border-radius:16px;padding:1rem 1.1rem;margin-bottom:.8rem}
[data-testid="stExpander"]{background:#111;border:1px solid var(--line);border-radius:12px}
[data-testid="stAlert"]{background:#171717;border:1px solid var(--line);border-radius:10px}
[data-testid="stChatInput"]{border-color:#3A3430!important}
[data-testid="stChatInput"] textarea{color:var(--white)!important}
hr{border-color:var(--line)!important}
[data-testid="stCaptionContainer"]{color:var(--muted)!important}
footer,#MainMenu{visibility:hidden}
@media(max-width:768px){.block-container{padding-top:1.4rem}.hero-title{font-size:2.65rem}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="brand-kicker">PRIVATE KNOWLEDGE WORKSPACE / 01</div>
<div class="hero-title">Your documents.<br><span>Your intelligence.</span></div>
<div class="hero-copy">A focused space to search, understand and ask questions across your own files.
Powered by local embeddings, hybrid retrieval and Ollama.</div>
<div class="orange-rule"></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div class="brand-kicker">PERSONAL RAG</div>
    <h2 style="font-size:1.55rem;margin:0;">Workspace<span style="color:#FF5A1F">.</span></h2>
    <p style="color:#A7A39B;font-size:.82rem;">Your private document library</p>
    <div class="section-label">System status</div>
    """, unsafe_allow_html=True)
    ready, status = check_ollama()
    if ready:
        st.success("OLLAMA ONLINE")
    else:
        st.warning("OLLAMA NEEDS ATTENTION")
        st.caption(status)
    st.markdown(f'<div class="tiny-mono" style="line-height:1.9;margin:.5rem 0 1rem;">EMBEDDINGS / {EMBEDDING_MODEL}<br>GENERATION / {LLM_MODEL}</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="section-label">Add to your library</div>', unsafe_allow_html=True)
    files = st.file_uploader("Drop files here", type=["pdf","docx","txt"], accept_multiple_files=True, help="PDF, DOCX and TXT · 20 MB maximum per file", label_visibility="collapsed")
    st.markdown('<div class="upload-note">Bring in your notes, reports and research. Your files are indexed locally on this machine.</div>', unsafe_allow_html=True)
    if st.button("＋  Index documents", use_container_width=True):
        if not files:
            st.warning("Choose at least one document first.")
        else:
            for file in files:
                try:
                    with st.spinner(f"Indexing {file.name}..."):
                        result = ingest_document(file, USER_ID)
                    st.success(f"{result['filename']} · {result['total_chunks']} chunks")
                except Exception as exc:
                    st.error(f"{file.name}: {exc}")
    st.divider()
    st.markdown('<div class="section-label">Indexed documents</div>', unsafe_allow_html=True)
    library = get_user_documents(USER_ID)
    if not library:
        st.caption("Your library is empty. Add a document to get started.")
    else:
        for doc_id, info in library.items():
            c1,c2 = st.columns([4,1])
            c1.markdown(f"<div style='font-size:.82rem;padding:.35rem 0;color:#F4F1EA'>▤ &nbsp;{info['filename']}</div>", unsafe_allow_html=True)
            if c2.button("×", key=f"delete_{doc_id}", help="Delete document"):
                try:
                    count = delete_document(doc_id, USER_ID)
                    st.toast(f"Removed document and {count} chunks.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not delete document: {exc}")

library = get_user_documents(USER_ID)
metrics = [
    ("DOCUMENTS", str(len(library)), "IN YOUR LIBRARY"),
    ("RETRIEVAL", "HYBRID", "VECTOR + BM25"),
    ("PROCESSING", "LOCAL", "POWERED BY OLLAMA"),
]
cols = st.columns(3)
for col,(label,value,note) in zip(cols,metrics):
    accent = "metric-accent" if value in ("HYBRID","LOCAL") else ""
    col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {accent}">{value}</div><div class="tiny-mono">{note}</div></div>',unsafe_allow_html=True)

st.markdown('<div class="section-label">Ask your library</div>',unsafe_allow_html=True)
st.markdown("""
<div class="surface-card" style="margin-bottom:1rem">
<div style="font-family:Manrope,sans-serif;font-size:1.12rem;font-weight:700">What would you like to know?</div>
<div class="upload-note" style="margin-top:.3rem">Ask naturally. The system searches relevant passages across your indexed documents.</div>
</div>
""",unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("SOURCE MATERIAL"):
                for source in message["sources"]:
                    meta = source["metadata"]
                    location = meta.get("filename","Unknown document")
                    if meta.get("page_number") is not None:
                        location += f" · PAGE {meta['page_number']}"
                    st.markdown(f"**{location}**")
                    st.write(source["text"])
                    st.caption(f"RRF relevance score: {source.get('rrf_score',0):.5f}")

question = st.chat_input("Ask a question about your documents...")
if question:
    st.session_state.messages.append({"role":"user","content":question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your library..."):
                result = ask_question(question, USER_ID, top_k=5)
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("SOURCE MATERIAL"):
                    for source in result["sources"]:
                        meta = source["metadata"]
                        location = meta.get("filename","Unknown document")
                        if meta.get("page_number") is not None:
                            location += f" · PAGE {meta['page_number']}"
                        st.markdown(f"**{location}**")
                        st.write(source["text"])
                        st.caption(f"RRF relevance score: {source.get('rrf_score',0):.5f}")
            st.session_state.messages.append({"role":"assistant","content":result["answer"],"sources":result["sources"]})
        except Exception as exc:
            error = f"Something went wrong: {exc}"
            st.error(error)
            st.session_state.messages.append({"role":"assistant","content":error})

st.markdown("""
<div style="border-top:1px solid #292929;margin-top:3rem;padding-top:1rem;display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap">
<span class="tiny-mono">PERSONAL RAG / DOCUMENT INTELLIGENCE</span>
<span class="tiny-mono">PRIVATE BY DESIGN · LOCAL AI</span>
</div>
""",unsafe_allow_html=True)
