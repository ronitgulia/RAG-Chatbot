"""
RAG Chatbot — Iron Man UI.
White background, Orbitron + Rajdhani fonts, modal-based config, settings popover.
"""
import streamlit as st
import logging
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from rag_pipeline import RAGPipeline
from web_scraper import WebScraper
from export_utils import export_as_txt, export_as_csv, export_as_json, export_as_pdf
from llm_provider import GROQ_MODELS
import styles  # injects CSS

logging.basicConfig(level=logging.INFO)


# ═══════════════════════════════════════════════════════════════════ #
#  Session State                                                       #
# ═══════════════════════════════════════════════════════════════════ #

def init_state():
    defaults = {
        "pipeline": None,
        "messages": [],
        "llm_ready": False,
        "scraper": WebScraper(),
        "search_mode": "hybrid",
        "top_k": 5,
        "multilingual": True,
        "eval_enabled": True,
        "eval_history": [],
        "doc_metadata": [],
        "show_llm_modal": False,
        "chunk_size": 512,
        "chunk_overlap": 64,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def get_pipeline() -> RAGPipeline:
    if st.session_state.pipeline is None:
        st.session_state.pipeline = RAGPipeline()
        # Sync session-state chunking values into the newly-created chunker
        # so files processed *before* the user opens the Settings panel use
        # the correct values.
        st.session_state.pipeline.update_chunker_settings(
            st.session_state.chunk_size,
            st.session_state.chunk_overlap,
        )
        # If the vector store was restored from cache, rebuild doc_metadata
        # so the "Indexed Documents" table repopulates after a page refresh.
        if st.session_state.pipeline.vector_store.is_ready and not st.session_state.doc_metadata:
            for src in st.session_state.pipeline.vector_store.get_document_sources():
                st.session_state.doc_metadata.append({
                    "name": src,
                    "size_kb": "—",
                    "uploaded_at": "restored from cache",
                })
    return st.session_state.pipeline


# ═══════════════════════════════════════════════════════════════════ #
#  Top Navigation Bar (HTML)                                           #
# ═══════════════════════════════════════════════════════════════════ #

llm_status_class = "connected" if st.session_state.llm_ready else "disconnected"
llm_status_text  = "LLM ONLINE"  if st.session_state.llm_ready else "LLM OFFLINE"

st.markdown(f"""
<div class="topbar">
  <div class="topbar-brand">RAG<span>BOT</span></div>
  <div class="topbar-right">
    <div class="made-in-india">
      <svg width="14" height="10" viewBox="0 0 14 10" style="border-radius:1px;overflow:hidden">
        <rect width="14" height="3.33" fill="#FF9933"/>
        <rect y="3.33" width="14" height="3.33" fill="#FFFFFF"/>
        <rect y="6.67" width="14" height="3.33" fill="#138808"/>
        <circle cx="7" cy="5" r="1.2" fill="none" stroke="#000080" stroke-width="0.4"/>
      </svg>
      Made in India
    </div>
    <span class="status-dot {llm_status_class}" title="{llm_status_text}"></span>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════ #
#  LLM Config Modal                                                    #
# ═══════════════════════════════════════════════════════════════════ #

@st.dialog("Configure Language Model")
def show_llm_modal():
    st.markdown('<p class="section-title">LLM Configuration</p>', unsafe_allow_html=True)

    provider = st.selectbox(
        "Provider", ["groq", "huggingface", "ollama"],
        help="Groq is free and fast. Ollama runs locally for fully offline RAG."
    )

    if provider == "groq":
        model = st.selectbox("Model", GROQ_MODELS)
    elif provider == "ollama":
        from llm_provider import OLLAMA_MODELS
        model = st.selectbox("Model", OLLAMA_MODELS)
    else:
        model = st.text_input("HuggingFace Model ID",
                              value="mistralai/Mistral-7B-Instruct-v0.2")

    if provider != "ollama":
        api_key = st.text_input(
            "API Key",
            type="password",
            placeholder="Paste your API key here (starts with gsk_...)" if provider == "groq"
                        else "Paste your HuggingFace token here"
        )
    else:
        api_key = "not-required"
        st.info("Ensure the Ollama app is running locally on your machine.")

    if provider == "groq":
        st.caption("Get a free API key at console.groq.com")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Connect", type="primary", use_container_width=True):
            if provider != "ollama" and not api_key.strip():
                st.error("API key is required.")
            else:
                with st.spinner("Connecting to LLM..."):
                    try:
                        p = get_pipeline()
                        p.llm.initialize(
                            provider=provider,
                            model_name=model,
                            api_key=api_key.strip() if provider != "ollama" else None
                        )
                        p.multilingual.enabled = st.session_state.multilingual
                        st.session_state.llm_ready = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Connection failed: {e}")
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


# Show a prominent prompt if LLM not connected (no auto-dialog to avoid rerun loops)
if not st.session_state.llm_ready:
    pass  # Prompt shown inline in the chat tab and via the "Configure LLM" button


# ═══════════════════════════════════════════════════════════════════ #
#  Settings Popover (top right area below topbar)                      #
# ═══════════════════════════════════════════════════════════════════ #

# Push content below fixed topbar
st.markdown('<div style="height:70px"></div>', unsafe_allow_html=True)

# Prominent LLM connection banner
if not st.session_state.llm_ready:
    st.warning("⚡ **Language Model not connected.** Click **Configure LLM** below to get started.")

# Settings button row
btn_col1, btn_col2, btn_col3 = st.columns([8, 1, 1])
with btn_col1:
    st.markdown(
        f'<div style="padding-top:8px; color:#666; font-size:0.95em;">'
        f'<b>Active Settings:</b> Mode: <span style="color:#2980B9">{st.session_state.search_mode.upper()}</span> '
        f'| Top-K: <span style="color:#2980B9">{st.session_state.top_k}</span> '
        f'| Chunk: <span style="color:#2980B9">{st.session_state.chunk_size}/{st.session_state.chunk_overlap}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
with btn_col2:
    if st.button("Configure LLM", key="btn_open_llm"):
        show_llm_modal()
with btn_col3:
    settings_open = st.button("Settings", key="btn_settings")

if settings_open:
    st.session_state["settings_panel"] = not st.session_state.get("settings_panel", False)

if st.session_state.get("settings_panel", False):
    with st.container():
        st.markdown("---")
        scol1, scol2, scol3 = st.columns(3)

        with scol1:
            st.markdown('<p class="section-title">Retrieval Settings</p>', unsafe_allow_html=True)
            st.session_state.search_mode = st.radio(
                "Search Mode", ["hybrid", "dense", "sparse"],
                horizontal=True, key="s_mode"
            )
            st.session_state.top_k = st.slider("Top-K Chunks", 1, 15, 5, key="s_topk")
            st.session_state.eval_enabled = st.toggle("Enable Evaluation", value=True)
            st.session_state.multilingual = st.toggle("Multilingual Mode", value=True)

        with scol2:
            st.markdown('<p class="section-title">Chunking</p>', unsafe_allow_html=True)

            def _sync_chunking():
                """Called on every slider / selectbox change so settings are always live."""
                st.session_state.chunk_size = st.session_state.s_cs
                st.session_state.chunk_overlap = st.session_state.s_co
                get_pipeline().update_chunker_settings(
                    st.session_state.s_cs,
                    st.session_state.s_co,
                    st.session_state.s_str,
                )

            st.slider(
                "Chunk Size", 128, 2048,
                value=st.session_state.chunk_size,
                step=64, key="s_cs", on_change=_sync_chunking,
            )
            st.slider(
                "Overlap", 0, 256,
                value=st.session_state.chunk_overlap,
                step=16, key="s_co", on_change=_sync_chunking,
            )
            st.selectbox(
                "Strategy", ["recursive", "character", "token"],
                key="s_str", on_change=_sync_chunking,
            )
            st.caption("✅ Settings apply automatically when changed.")

        with scol3:
            st.markdown('<p class="section-title">Actions & Status</p>', unsafe_allow_html=True)
            p = get_pipeline()
            status = p.status
            st.metric("Indexed Chunks", status["documents_indexed"])
            st.metric("Chat Turns", status["conversation_turns"])

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Clear Chat", key="s_clr", use_container_width=True):
                    st.session_state.messages = []
                    p.reset_conversation()
                    st.rerun()
            with c2:
                if st.button("Reset All", key="s_rst", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.eval_history = []
                    st.session_state.doc_metadata = []
                    p.reset_all()
                    st.rerun()

            if status["sources"]:
                st.markdown("**Sources:**")
                for s in status["sources"]:
                    st.markdown(f"`{s.split('/')[-1]}`")

        st.markdown("---")


# ═══════════════════════════════════════════════════════════════════ #
#  Main Tabs                                                           #
# ═══════════════════════════════════════════════════════════════════ #

tab_chat, tab_docs, tab_eval, tab_export = st.tabs(
    ["Chat", "Documents", "Evaluation", "Export"]
)


# ── Chat ────────────────────────────────────────────────────────────
with tab_chat:
    if not st.session_state.llm_ready:
        st.info("Click 'Configure LLM' above to connect your language model.")
    if not get_pipeline().vector_store.is_ready:
        st.info("No documents indexed. Go to the Documents tab to add knowledge.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            meta = []
            if msg.get("timestamp"):   meta.append(msg["timestamp"])
            if msg.get("sources"):     meta.append("Sources: " + ", ".join(
                s.split("/")[-1] for s in msg["sources"]))
            if msg.get("retrieval_mode"): meta.append(f"Mode: {msg['retrieval_mode']}")
            if meta:
                st.caption("  |  ".join(meta))

            ev = msg.get("eval")
            if ev:
                with st.expander("Evaluation Scores"):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Faithfulness",  f"{ev.faithfulness:.2f}")
                    c2.metric("Relevancy",     f"{ev.answer_relevancy:.2f}")
                    c3.metric("Precision",     f"{ev.context_precision:.2f}")
                    c4.metric("Overall",       f"{ev.overall_score:.2f}")

            if msg.get("chunks") and msg["role"] == "assistant":
                with st.expander(f"Retrieved Chunks ({len(msg['chunks'])})"):
                    for i, chunk in enumerate(msg["chunks"]):
                        src = chunk.get("metadata", {}).get("source", "Unknown")
                        pg  = chunk.get("metadata", {}).get("page", "")
                        pg_str = f" · Page {pg}" if pg else ""
                        content = chunk.get("page_content", "")
                        display_text = content[:400] + ("..." if len(content) > 400 else "")
                        st.markdown(
                            f'<div class="chunk-card">'
                            f'<span class="chunk-source">[{i+1}] {src}{pg_str}</span>'
                            f'<p class="chunk-text">{display_text}</p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    p = get_pipeline()
    user_input = st.chat_input(
        "Ask a question about your documents...",
        disabled=not (p.llm.is_ready and p.vector_store.is_ready),
    )

    if user_input:
        ts = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({"role": "user", "content": user_input, "timestamp": ts})

        with st.spinner("Retrieving and generating..."):
            result = p.query(
                question=user_input,
                top_k=st.session_state.top_k,
                search_mode=st.session_state.search_mode,
                evaluate=st.session_state.eval_enabled,
            )

        ev = result.get("evaluation")
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "sources": result.get("sources", []),
            "chunks": result.get("chunks", []),
            "eval": ev,
            "retrieval_mode": result.get("retrieval_mode", ""),
        })

        if ev:
            st.session_state.eval_history.append({
                "question": user_input,
                "answer": result["answer"],
                "faithfulness": ev.faithfulness,
                "answer_relevancy": ev.answer_relevancy,
                "context_precision": ev.context_precision,
                "overall_score": ev.overall_score,
                "timestamp": ts,
            })
        st.rerun()


# ── Documents ────────────────────────────────────────────────────────
with tab_docs:
    st.markdown('<p class="section-title">Document Management</p>', unsafe_allow_html=True)
    col_up, col_web = st.columns(2, gap="large")

    with col_up:
        st.markdown("**Upload Files**")
        st.caption("Supported: PDF, DOCX, TXT, Markdown")
        uploaded = st.file_uploader(
            "Drop files here", type=["pdf","docx","doc","txt","md"],
            accept_multiple_files=True, label_visibility="collapsed"
        )
        if uploaded:
            if st.button("Process Files", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                total_files = len(uploaded)
                
                success_count = 0
                total_chunks = 0
                all_errors = []
                
                for i, file in enumerate(uploaded):
                    status_text.text(f"Processing {file.name} ({i+1}/{total_files})...")
                    result = get_pipeline().ingest_files([file])
                    
                    if result["success"]:
                        success_count += 1
                        total_chunks += result.get("chunks_added", 0)
                        st.session_state.doc_metadata.append({
                            "name": file.name,
                            "size_kb": round(file.size / 1024, 1),
                            "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        })
                    if result.get("errors"):
                        all_errors.extend(result["errors"])
                        
                    progress_bar.progress((i + 1) / total_files)
                
                status_text.empty()
                progress_bar.empty()
                
                if success_count > 0:
                    st.success(f"Indexed {total_chunks} chunks from {success_count} file(s).")
                if all_errors:
                    st.warning("Errors: " + "; ".join(all_errors))
                if success_count == 0 and not all_errors:
                    st.error("Failed to process files.")

    with col_web:
        st.markdown("**Web Scraping**")
        urls_input = st.text_area("URLs (one per line)",
            placeholder="https://example.com", height=120,
            label_visibility="collapsed")
        follow = st.checkbox("Follow internal links")
        max_p  = st.number_input("Max pages", 1, 20, 5)
        if st.button("Scrape and Index", use_container_width=True):
            from urllib.parse import urlparse
            raw_urls = [u.strip() for u in urls_input.strip().splitlines() if u.strip()]
            urls = []
            invalid_urls = []
            for u in raw_urls:
                parsed = urlparse(u)
                if parsed.scheme in ("http", "https") and parsed.netloc:
                    urls.append(u)
                else:
                    invalid_urls.append(u)
            
            if invalid_urls:
                st.warning(f"Ignored invalid or non-HTTP URLs: {', '.join(invalid_urls)}")
            
            if not urls:
                if not invalid_urls:
                    st.warning("Enter at least one URL.")
            else:
                with st.spinner(f"Scraping {len(urls)} URL(s)..."):
                    docs = st.session_state.scraper.scrape_multiple(urls, follow_links=follow, max_pages=max_p)
                if not docs:
                    st.error("No content scraped.")
                else:
                    r = get_pipeline().ingest_web_documents(docs)
                    if r["success"]:
                        st.success(f"Indexed {r['chunks_added']} chunks from {len(docs)} page(s).")
                        for doc in docs:
                            src = doc.get("metadata", {}).get("source", "Unknown URL")
                            size_kb = round(len(doc.get("page_content", "")) / 1024, 1)
                            st.session_state.doc_metadata.append({
                                "name": src,
                                "size_kb": size_kb,
                                "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            })
                    else:
                        st.error(r["message"])

    st.markdown("---")
    st.markdown("**Wikipedia Integration**")
    wiki_col1, wiki_col2 = st.columns([3, 1])
    with wiki_col1:
        wiki_topic = st.text_input("Enter Wikipedia Topic", placeholder="e.g. Albert Einstein", label_visibility="collapsed")
    with wiki_col2:
        fetch_wiki = st.button("Fetch & Summarize", use_container_width=True)
        
    if fetch_wiki:
        if not wiki_topic.strip():
            st.warning("Please enter a topic.")
        else:
            with st.spinner(f"Fetching Wikipedia page for '{wiki_topic}'..."):
                try:
                    import wikipedia
                    page = wikipedia.page(wiki_topic, auto_suggest=True)
                    content = page.content
                    title = page.title
                    url = page.url
                    
                    docs = [{
                        "page_content": content,
                        "metadata": {
                            "source": url,
                            "title": title,
                            "file_type": "wikipedia",
                        }
                    }]
                    r = get_pipeline().ingest_web_documents(docs)
                    
                    if r["success"]:
                        st.success(f"Indexed {r['chunks_added']} chunks from Wikipedia: {title}")
                        st.session_state.doc_metadata.append({
                            "name": url,
                            "size_kb": round(len(content) / 1024, 1),
                            "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        })
                        
                        p = get_pipeline()
                        if p.llm.is_ready:
                            with st.spinner("Generating full detail analysis and summary..."):
                                result = p.query(
                                    question=f"Provide a full detail analysis and comprehensive summary of the topic: {title}. Focus on the most important information.",
                                    top_k=15,
                                    search_mode="hybrid",
                                    evaluate=False
                                )
                                st.markdown("### Wikipedia Analysis & Summary")
                                st.info(result["answer"])
                                
                                ts = datetime.now().strftime("%H:%M:%S")
                                st.session_state.messages.append({"role": "user", "content": f"Fetch and summarize Wikipedia page for: {title}", "timestamp": ts})
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": result["answer"], 
                                    "timestamp": ts, 
                                    "sources": result.get("sources", []),
                                    "chunks": result.get("chunks", [])
                                })
                        else:
                            st.warning("LLM not configured. Content indexed, but summary skipped. Please configure LLM in Settings.")
                    else:
                        st.error(r["message"])
                except ImportError:
                    st.error("The 'wikipedia' Python package is not installed. Please install it using `pip install wikipedia`.")
                except Exception as e:
                    if "DisambiguationError" in str(type(e)):
                        st.error(f"Topic is ambiguous. Try a more specific term.")
                    elif "PageError" in str(type(e)):
                        st.error(f"No Wikipedia page found for '{wiki_topic}'.")
                    else:
                        st.error(f"Error fetching Wikipedia content: {e}")

    if st.session_state.doc_metadata:
        st.markdown("---")
        st.markdown("**Indexed Documents**")
        st.dataframe(pd.DataFrame(st.session_state.doc_metadata),
                     use_container_width=True, hide_index=True)
        if st.button("Clear All Documents"):
            st.session_state.doc_metadata = []
            get_pipeline().reset_all()
            st.rerun()


# ── Evaluation ───────────────────────────────────────────────────────
with tab_eval:
    st.markdown('<p class="section-title">Evaluation Dashboard</p>', unsafe_allow_html=True)

    if not st.session_state.eval_history:
        st.info("No evaluation data yet. Start chatting to collect metrics.")
    else:
        df = pd.DataFrame(st.session_state.eval_history)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Faithfulness",    f"{df['faithfulness'].mean():.3f}")
        c2.metric("Avg Relevancy",       f"{df['answer_relevancy'].mean():.3f}")
        c3.metric("Avg Precision",       f"{df['context_precision'].mean():.3f}")
        c4.metric("Avg Overall",         f"{df['overall_score'].mean():.3f}")

        st.markdown("---")
        ca, cb = st.columns(2)
        with ca:
            fig = px.line(df.reset_index(), x="index",
                y=["faithfulness","answer_relevancy","context_precision","overall_score"],
                color_discrete_sequence=["#C0392B","#E67E22","#27AE60","#2980B9"],
                labels={"index":"Query #","value":"Score","variable":"Metric"})
            fig.update_layout(plot_bgcolor="#fff", paper_bgcolor="#fff",
                              font_family="Rajdhani", margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(fig, use_container_width=True)
        with cb:
            fig2 = go.Figure()
            for col, col_color in zip(
                ["faithfulness","answer_relevancy","context_precision"],
                ["#C0392B","#E67E22","#27AE60"]
            ):
                fig2.add_trace(go.Box(y=df[col],
                    name=col.replace("_"," ").title(),
                    marker_color=col_color, boxpoints="all"))
            fig2.update_layout(plot_bgcolor="#fff", paper_bgcolor="#fff",
                               font_family="Rajdhani", margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(df[["timestamp","question","faithfulness",
                          "answer_relevancy","context_precision","overall_score"]],
                     use_container_width=True, hide_index=True)

        if st.button("Clear Evaluation History"):
            st.session_state.eval_history = []
            st.rerun()


# ── Export ───────────────────────────────────────────────────────────
with tab_export:
    st.markdown('<p class="section-title">Export</p>', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.info("No messages to export. Start a conversation first.")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        msgs = st.session_state.messages
        ecol1, ecol2 = st.columns(2)

        with ecol1:
            st.markdown("**Download Chat History**")
            fmt = st.selectbox("Format", ["TXT","CSV","JSON","PDF"])
            if st.button("Generate Export", use_container_width=True):
                try:
                    if fmt == "TXT":
                        data, mime, ext = export_as_txt(msgs, "RAG Chatbot Export"), "text/plain", "txt"
                    elif fmt == "CSV":
                        data, mime, ext = export_as_csv(msgs), "text/csv", "csv"
                    elif fmt == "JSON":
                        data, mime, ext = export_as_json(msgs), "application/json", "json"
                    else:
                        data = export_as_pdf(msgs, "RAG Chatbot Export")
                        # export_as_pdf falls back to TXT internally when
                        # fpdf2 is missing — detect that via a quick heuristic
                        # (PDF bytes always start with %PDF).
                        if data and data[:4] == b"%PDF":
                            mime, ext = "application/pdf", "pdf"
                        else:
                            mime, ext = "text/plain", "txt"
                            st.warning("PDF library not available — falling back to TXT.")
                except Exception as e:
                    st.error(f"Export failed: {e}")
                    data, mime, ext = export_as_txt(msgs, "RAG Chatbot Export"), "text/plain", "txt"

                st.download_button(f"Download .{ext}", data=data,
                    file_name=f"rag_chat_{ts}.{ext}", mime=mime)

        with ecol2:
            st.markdown("**Evaluation Report**")
            if st.session_state.eval_history:
                csv_b = pd.DataFrame(st.session_state.eval_history).to_csv(index=False).encode("utf-8")
                st.download_button("Download Evaluation CSV", data=csv_b,
                    file_name=f"rag_eval_{ts}.csv", mime="text/csv",
                    use_container_width=True)
            else:
                st.info("No evaluation data to export.")


# ── Footer ───────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer-strip">'
    'Built with LangChain · FAISS · RAGAS · Groq &nbsp;&nbsp;|&nbsp;&nbsp; '
    '<span>Made in India</span>'
    '</div>',
    unsafe_allow_html=True
)