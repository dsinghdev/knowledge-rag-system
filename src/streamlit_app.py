"""
Income Tax RAG Assistant — Streamlit Chat UI.

This frontend ONLY imports backend_service, keeping a clean boundary.
To decouple later, replace the backend_service import with HTTP calls.
"""

import streamlit as st
import backend_service


# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Income Tax RAG Assistant",
    page_icon="💰",
    layout="centered",
)

# ── Custom styles ────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        max-width: 820px;
        padding-top: 2rem;
    }

    /* Header gradient */
    .app-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 50%, #3b82f6 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
    }
    .app-header h1 {
        margin: 0;
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .app-header p {
        margin: 0.4rem 0 0 0;
        opacity: 0.85;
        font-size: 0.95rem;
    }

    /* Source badge */
    .source-badge {
        display: inline-block;
        background: #e0f2fe;
        color: #0369a1;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 999px;
        margin: 0 3px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ───────────────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
        <h1>💰 Income Tax RAG Assistant</h1>
        <p>Ask questions about Indian Income Tax — powered by official documents</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("### ℹ️ About")
    st.markdown(
        "This assistant answers Indian Income Tax queries using "
        "**Retrieval-Augmented Generation (RAG)**.\n\n"
        "Responses are grounded in official tax documents — "
        "the model will not make up information."
    )
    st.divider()
    st.caption("Built with Streamlit · LangChain · Gemini")

# ── Chat state ───────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# ── Chat input ───────────────────────────────────────────
if prompt := st.chat_input("Ask a question about Income Tax…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer from backend
    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            result = backend_service.ask(prompt)

        # Build response with source badges
        answer_text = result["answer"]
        if result["sources"]:
            badges = " ".join(
                f'<span class="source-badge">{s}</span>'
                for s in result["sources"]
            )
            answer_text += f"\n\n**Sources:** {badges}"

        st.markdown(answer_text, unsafe_allow_html=True)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer_text}
        )
