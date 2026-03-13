"""
app.py
─────────────────────────────────────────────────────────────────────────────
NeoStats — AI-Powered Investment Research Assistant
Streamlit UI
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import os
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# ── project imports ──────────────────────────────────────────────────────────
from config.config import (
    CONVERSATION_WINDOW,
    RESPONSE_MODE_CONCISE,
    RESPONSE_MODE_DETAILED,
)
from utils.db_manager import DatabaseManager
from utils.intent_sql_engine import (
    ClarificationManager,
    IntentClassifier,
    EntityResolver,
    ResponseFormatter,
    SQLGenerator,
    SourceType,
    triage_query,
)
from utils.rag_engine import RAGEngine
from utils.web_search import web_answer
from utils.synthesizer import combine_answers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="NeoStats | Investment Research Assistant",
    page_icon="📊",
    layout="wide",
)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────

def _init_state():
    defaults = {
        "chat_history":      [],        # list of {role, content}
        "conv_context":      [],        # list of {query, response} for LLM context
        "db_manager":        None,
        "rag_engine":        RAGEngine(),
        "classifier":        IntentClassifier(),
        "clarification_mgr": ClarificationManager(),
        "formatter":         ResponseFormatter(),
        "db_ready":          False,
        "response_mode":     RESPONSE_MODE_CONCISE,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/bar-chart.png", width=64)
    st.title("NeoStats")
    st.caption("AI-Powered Investment Research Assistant")
    st.divider()

    # ── Dataset upload ───────────────────────────────────────────────────────
    st.subheader("📂 Portfolio Datasets")
    col1, col2 = st.columns(2)
    holdings_file = col1.file_uploader("holdings.csv", type=["csv"], key="holdings_upload")
    trades_file   = col2.file_uploader("trades.csv",   type=["csv"], key="trades_upload")

    if holdings_file and trades_file:
        if st.button("🔄 Load Datasets", use_container_width=True):
            with st.spinner("Initialising database…"):
                try:
                    os.makedirs("data", exist_ok=True)
                    h_path = f"data/holdings_{int(time.time())}.csv"
                    t_path = f"data/trades_{int(time.time())}.csv"
                    with open(h_path, "wb") as f: f.write(holdings_file.read())
                    with open(t_path, "wb") as f: f.write(trades_file.read())

                    db = DatabaseManager()
                    h_df, t_df = db.init_from_csvs(h_path, t_path)
                    st.session_state["db_manager"] = db
                    st.session_state["db_ready"]   = True
                    st.success(f"✅ {len(h_df):,} holdings & {len(t_df):,} trades loaded")

                    # Preview
                    with st.expander("Holdings preview"):
                        st.dataframe(h_df.head(5), use_container_width=True)
                    with st.expander("Trades preview"):
                        st.dataframe(t_df.head(5), use_container_width=True)
                except Exception as exc:
                    st.error(f"Dataset load failed: {exc}")

    st.divider()

    # ── Research docs upload (RAG) ───────────────────────────────────────────
    st.subheader("📄 Research Documents (RAG)")
    doc_files = st.file_uploader(
        "Upload PDFs or DOCX files",
        type=["pdf", "docx", "doc"],
        accept_multiple_files=True,
        key="doc_upload",
    )
    if doc_files and st.button("📥 Ingest Documents", use_container_width=True):
        rag: RAGEngine = st.session_state["rag_engine"]
        for doc in doc_files:
            with st.spinner(f"Ingesting {doc.name}…"):
                try:
                    suffix = Path(doc.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(doc.read())
                        tmp_path = tmp.name
                    n_chunks = rag.ingest(doc.name, tmp_path)
                    os.unlink(tmp_path)
                    if n_chunks:
                        st.success(f"✅ {doc.name}: {n_chunks} chunks indexed")
                    else:
                        st.warning(f"⚠️ {doc.name}: no text extracted")
                except Exception as exc:
                    st.error(f"Ingest failed ({doc.name}): {exc}")

    if st.session_state["rag_engine"].index_names:
        st.caption("Indexed: " + ", ".join(st.session_state["rag_engine"].index_names))

    st.divider()

    # ── Settings ─────────────────────────────────────────────────────────────
    st.subheader("⚙️ Settings")
    mode = st.radio(
        "Response mode",
        [RESPONSE_MODE_CONCISE, RESPONSE_MODE_DETAILED],
        horizontal=True,
    )
    st.session_state["response_mode"] = mode

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state["chat_history"]  = []
        st.session_state["conv_context"]  = []
        st.session_state["clarification_mgr"].clear()
        st.rerun()

    st.divider()
    st.caption("ℹ️ Data sources used per query are shown in the chat.")

# ─────────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────────

st.title("📊 NeoStats — Investment Research Assistant")

# Status bar
col_a, col_b, col_c = st.columns(3)
col_a.metric("Database",    "✅ Ready" if st.session_state["db_ready"] else "⚠️ Not loaded")
col_b.metric("RAG indexes", str(len(st.session_state["rag_engine"].index_names)))
col_c.metric("Mode",        st.session_state["response_mode"].title())

st.divider()

# ── Render chat history ──────────────────────────────────────────────────────
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ───────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about your portfolio, research docs, or market news…")

if user_input:
    # Show user message
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ── Generate response ────────────────────────────────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                clarif_mgr:  ClarificationManager = st.session_state["clarification_mgr"]
                classifier:  IntentClassifier      = st.session_state["classifier"]
                formatter:   ResponseFormatter     = st.session_state["formatter"]
                rag:         RAGEngine             = st.session_state["rag_engine"]
                db_manager:  DatabaseManager | None = st.session_state["db_manager"]
                conv_ctx                           = st.session_state["conv_context"]
                mode                               = st.session_state["response_mode"]

                # ── CLARIFICATION PENDING ────────────────────────────────────
                if clarif_mgr.has_pending:
                    intent = clarif_mgr.resolve(user_input)
                    if intent is None:
                        answer = (
                            f"I still need clarification: **{clarif_mgr.question}**\n\n"
                            "Please reply with 'holdings', 'trades', or 'both'."
                        )
                        st.markdown(answer)
                        st.session_state["chat_history"].append(
                            {"role": "assistant", "content": answer}
                        )
                        st.stop()
                else:
                    intent = None

                # ── TRIAGE ───────────────────────────────────────────────────
                source = triage_query(user_input, conv_ctx[-CONVERSATION_WINDOW:])

                # ── PROACTIVE ENTITY PRE-CHECK ───────────────────────────────
                # Before trusting the triage result, scan the raw query for
                # known portfolio names (even misspelled). If one is found,
                # force SQL routing so the query always hits the database.
                _resolved_portfolio: str | None = None
                if db_manager and st.session_state["db_ready"]:
                    try:
                        _pre_resolver = EntityResolver(db_manager.conn)
                        _resolved_portfolio = _pre_resolver.find_portfolio_in_query(user_input)
                        if _resolved_portfolio and source not in (
                            SourceType.SQL, SourceType.SQL_RAG,
                            SourceType.SQL_WEB, SourceType.ALL,
                        ):
                            logger.info(
                                "Proactive entity check: '%s' matched portfolio '%s'; "
                                "overriding triage source to SQL.",
                                user_input, _resolved_portfolio,
                            )
                            source = SourceType.SQL
                    except Exception as _pre_exc:
                        logger.warning("Proactive entity pre-check error: %s", _pre_exc)

                badge_map = {
                    SourceType.SQL:     "🗄️ SQL",
                    SourceType.RAG:     "📄 RAG",
                    SourceType.WEB:     "🌐 Web",
                    SourceType.SQL_RAG: "🗄️ SQL + 📄 RAG",
                    SourceType.SQL_WEB: "🗄️ SQL + 🌐 Web",
                    SourceType.RAG_WEB: "📄 RAG + 🌐 Web",
                    SourceType.ALL:     "🗄️ SQL + 📄 RAG + 🌐 Web",
                }
                st.caption(f"Sources: {badge_map.get(source, str(source))}")

                needs_sql = source in (
                    SourceType.SQL, SourceType.SQL_RAG,
                    SourceType.SQL_WEB, SourceType.ALL,
                )
                needs_rag = source in (
                    SourceType.RAG, SourceType.SQL_RAG,
                    SourceType.RAG_WEB, SourceType.ALL,
                )
                needs_web = source in (
                    SourceType.WEB, SourceType.SQL_WEB,
                    SourceType.RAG_WEB, SourceType.ALL,
                )

                sql_answer = rag_answer = web_answer_text = None

                # ── SQL PATH ─────────────────────────────────────────────────
                if needs_sql:
                    if db_manager is None or not st.session_state["db_ready"]:
                        sql_answer = "*(Portfolio database not loaded — please upload CSVs in the sidebar.)*"
                    else:
                        if intent is None:
                            intent = classifier.classify(
                                user_input, conv_ctx[-CONVERSATION_WINDOW:]
                            )

                        if intent.needs_clarification:
                            clarif_mgr.set_pending(intent, user_input)
                            q = intent.clarification_question or "Could you clarify your question?"
                            answer = f"🤔 **Clarification needed:**\n\n{q}"
                            st.markdown(answer)
                            st.session_state["chat_history"].append(
                                {"role": "assistant", "content": answer}
                            )
                            st.stop()

                        # Inject pre-resolved portfolio name (from entity pre-check above).
                        # If the proactive check already matched a portfolio, use it directly
                        # instead of running a second LLM call via EntityResolver.
                        if _resolved_portfolio:
                            if intent.filters is None:
                                intent.filters = {}
                            # Only inject if classifier didn't set a better value
                            existing = intent.filters.get("PortfolioName", "")
                            if not existing or existing.lower() != _resolved_portfolio.lower():
                                intent.filters["PortfolioName"] = _resolved_portfolio
                        elif intent.filters and "PortfolioName" in intent.filters:
                            # Fallback: fuzzy-resolve whatever the classifier gave us
                            resolver = EntityResolver(db_manager.conn)
                            resolved = resolver.resolve_portfolio(intent.filters["PortfolioName"])
                            if resolved:
                                intent.filters["PortfolioName"] = resolved

                        # Build & run SQL
                        sql_gen = SQLGenerator(db_manager.conn)
                        sql     = sql_gen.generate(intent)

                        if sql:
                            with st.expander("🔍 Generated SQL (deterministic, no LLM)"):
                                st.code(sql, language="sql")
                            ok, rows, err = db_manager.execute(sql)
                            if ok:
                                sql_answer = formatter.format(user_input, rows, intent, mode)
                                if rows:
                                    df = pd.DataFrame(rows)
                                    with st.expander("📊 Raw Data"):
                                        st.dataframe(df, use_container_width=True)
                            else:
                                sql_answer = f"Query error: {err}"
                        else:
                            sql_answer = None

                # ── RAG PATH ─────────────────────────────────────────────────
                if needs_rag and rag.index_names:
                    rag_answer = rag.query(user_input, mode=mode)

                # ── WEB PATH ─────────────────────────────────────────────────
                if needs_web:
                    try:
                        web_answer_text = web_answer(user_input, mode=mode)
                    except Exception as exc:
                        logger.error("Web search error: %s", exc)
                        web_answer_text = "*(Web search unavailable.)*"

                # ── WEB FALLBACK ──────────────────────────────────────────
                def _is_empty(ans):
                    if not ans:
                        return True
                    low = ans.lower()
                    return any(phrase in low for phrase in [
                        "couldn't find", "not loaded", "no recent",
                        "not enough information", "query error",
                    ])

                if not needs_web and _is_empty(sql_answer) and _is_empty(rag_answer):
                    try:
                        web_answer_text = web_answer(user_input, mode=mode)
                        st.caption("🌐 Web search (fallback — SQL & RAG had no results)")
                    except Exception as exc:
                        logger.error("Web fallback error: %s", exc)

                # ── COMBINE ──────────────────────────────────────────────────
                answer = combine_answers(
                    user_input, sql_answer, rag_answer, web_answer_text, mode
                )

            except Exception as exc:
                logger.error("Chat handler error: %s", exc, exc_info=True)
                answer = f"An error occurred: {exc}"

        st.markdown(answer)

    # Store
    st.session_state["chat_history"].append({"role": "assistant", "content": answer})
    st.session_state["conv_context"].append({"query": user_input, "response": answer})

    # Keep context window bounded
    if len(st.session_state["conv_context"]) > CONVERSATION_WINDOW * 2:
        st.session_state["conv_context"] = st.session_state["conv_context"][-CONVERSATION_WINDOW:]
