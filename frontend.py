"""
app.py  —  Streamlit frontend for the Self-RAG chatbot
Run with:  streamlit run app.py
"""

import streamlit as st
from backend import run_rag

st.set_page_config(
    page_title="NexaAI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }

    .user-bubble {
        background: #EEF2FF;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 18px;
        margin: 6px 0 6px auto;
        max-width: 75%;
        color: #1e1b4b;
        font-size: 15px;
        line-height: 1.55;
        display: table;
        margin-left: auto;
    }
    .bot-bubble {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 18px;
        margin: 6px 0;
        max-width: 75%;
        color: #111827;
        font-size: 15px;
        line-height: 1.55;
    }
    div.stButton > button {
        width: 100%;
        text-align: left;
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 10px 14px;
        font-size: 13.5px;
        color: #374151;
        white-space: normal;
        height: auto;
    }
    div.stButton > button:hover {
        background: #EEF2FF;
        border-color: #818cf8;
        color: #1e1b4b;
    }
    .badge-green  { color:#166534; background:#dcfce7; border-radius:6px; padding:2px 9px; font-size:12px; font-weight:600; }
    .badge-yellow { color:#854d0e; background:#fef9c3; border-radius:6px; padding:2px 9px; font-size:12px; font-weight:600; }
    .badge-red    { color:#991b1b; background:#fee2e2; border-radius:6px; padding:2px 9px; font-size:12px; font-weight:600; }
    .trace-step {
        border-left: 3px solid #c7d2fe;
        padding: 4px 10px;
        margin: 3px 0;
        font-size: 13px;
        color: #374151;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

EXAMPLE_QUESTIONS = {
    "🏢 Company": [
        "What is NexaAI?",
        "Who is the CEO of NexaAI?",
        "Who founded NexaAI and when?",
        "Describe NexaAI's company culture.",
    ],
    "📦 Products": [
        "What products does NexaAI offer?",
        "What is NexaChat and what can it do?",
        "What integrations does NexaInsight support?",
    ],
    "💰 Pricing & Policies": [
        "What are NexaAI's pricing plans?",
        "Does NexaAI offer a free trial?",
        "What is the refund policy?",
        "What is the employee notice period policy?",
    ],
}

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_trace" not in st.session_state:
    st.session_state.last_trace = []
if "last_meta" not in st.session_state:
    st.session_state.last_meta = {}
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


def handle_question(question: str):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Thinking…"):
        result = run_rag(question)
    answer = result.get("answer", "I couldn't find an answer.")
    meta = {
        "issup":          result.get("issup", ""),
        "isuse":          result.get("isuse", ""),
        "use_reason":     result.get("use_reason", ""),
        "need_retrieval": result.get("need_retrieval", False),
        "retries":        result.get("retries", 0),
        "rewrite_tries":  result.get("rewrite_tries", 0),
        "relevant_docs":  result.get("relevant_docs", []),
        "evidence":       result.get("evidence", []),
    }
    st.session_state.last_trace = result.get("trace", [])
    st.session_state.last_meta  = meta
    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})


# ── Sidebar ──
with st.sidebar:
    st.markdown("### 🧠 RAG Trace")
    st.caption("Step-by-step reasoning from the last response")
    meta = st.session_state.last_meta

    if meta:
        col1, col2 = st.columns(2)
        with col1:
            issup = meta.get("issup", "")
            if issup == "fully_supported":
                st.markdown('<span class="badge-green">✔ Grounded</span>', unsafe_allow_html=True)
            elif issup == "partially_supported":
                st.markdown('<span class="badge-yellow">⚠ Partial</span>', unsafe_allow_html=True)
            elif issup == "no_support":
                st.markdown('<span class="badge-red">✘ Ungrounded</span>', unsafe_allow_html=True)
        with col2:
            isuse = meta.get("isuse", "")
            if isuse == "useful":
                st.markdown('<span class="badge-green">✔ Useful</span>', unsafe_allow_html=True)
            elif isuse == "not_useful":
                st.markdown('<span class="badge-red">✘ Not useful</span>', unsafe_allow_html=True)

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Retrieval", "✅" if meta.get("need_retrieval") else "⬜")
        c2.metric("Revisions", meta.get("retries", 0))
        c3.metric("Rewrites",  meta.get("rewrite_tries", 0))

        relevant_docs = meta.get("relevant_docs", [])
        if relevant_docs:
            st.markdown(f"**Chunks used:** {len(relevant_docs)}")
            with st.expander("📄 View sources"):
                seen = set()
                for doc in relevant_docs:
                    src  = (doc.metadata or {}).get("source", "unknown")
                    page = (doc.metadata or {}).get("page", None)
                    key  = f"{src}:{page}"
                    if key not in seen:
                        seen.add(key)
                        label = f"`{src}`" + (f" — p.{page}" if page is not None else "")
                        st.markdown(label)

        evidence = meta.get("evidence", [])
        if evidence:
            with st.expander("🔍 Supporting evidence"):
                for e in evidence:
                    st.markdown(f"> {e}")

        if meta.get("use_reason"):
            st.caption(f"💬 {meta['use_reason']}")

        st.markdown("---")

    if st.session_state.last_trace:
        for step in st.session_state.last_trace:
            st.markdown(f'<div class="trace-step">{step}</div>', unsafe_allow_html=True)
    else:
        st.caption("Ask a question to see the pipeline trace here.")

    st.markdown("---")
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.last_trace = []
        st.session_state.last_meta = {}
        st.session_state.pending_question = None
        st.rerun()


# ── Main area ──
st.markdown("## 🤖 NexaAI Assistant")
st.caption("Ask anything about NexaAI's products, policies, or company profile.")

# Welcome screen — shown only when no messages yet
if not st.session_state.messages:
    st.markdown("---")
    st.markdown("#### 💡 Try asking one of these")

    for category, questions in EXAMPLE_QUESTIONS.items():
        st.markdown(f"**{category}**")
        cols = st.columns(2)
        for i, q in enumerate(questions):
            if cols[i % 2].button(q, key=f"eq_{category}_{i}"):
                st.session_state.pending_question = q
                st.rerun()

    st.markdown("---")

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">🧑&nbsp; {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        content = msg["content"].replace("\n", "<br>")
        st.markdown(
            f'<div class="bot-bubble">🤖&nbsp; {content}</div>',
            unsafe_allow_html=True,
        )

# Fire pending question (from example cards)
if st.session_state.pending_question:
    q = st.session_state.pending_question
    st.session_state.pending_question = None
    handle_question(q)
    st.rerun()

# Chat input
if prompt := st.chat_input("Ask a question about NexaAI…"):
    handle_question(prompt)
    st.rerun()