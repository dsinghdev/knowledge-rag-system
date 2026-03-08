"""
Investment RAG Assistant — Streamlit Chat UI.

This frontend ONLY imports backend_service, keeping a clean boundary.
To decouple later, replace the backend_service import with HTTP calls.
"""

import streamlit as st
import sys
import os
import logging

logger = logging.getLogger(__name__)

# The container sets PYTHONPATH, but this helps for local execution
if os.path.dirname(__file__) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

try:
    import rag_engine
except ImportError:
    # Fallback for different envs
    from backend import rag_engine


# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Investment RAG Assistant",
    page_icon="📈",
    layout="centered",
)

# ── Custom styles ────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        max-width: 820px;
        padding-top: 2rem;
    }

    /* Header gradient - Investment theme */
    .app-header {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 50%, #0d9488 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
        overflow: hidden;
    }
    .app-header h1 {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: white !important;
    }
    .app-header p {
        margin: 0.4rem 0 0 0;
        opacity: 0.9;
        font-size: 0.88rem;
        color: white !important;
    }

    /* Source badge - works on both light and dark */
    .source-badge {
        display: inline-block;
        background: #065f46;
        color: #d1fae5;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 999px;
        margin: 0 3px;
        border: 1px solid #0d9488;
    }

    /* DO NOT force text colors — let Streamlit handle light/dark mode natively */

    /* Hide broken Material Icon text in sidebar toggle */
    button[kind="headerNoPadding"] {
        font-size: 0 !important;
    }
    [data-testid="collapsedControl"] {
        font-size: 0 !important;
    }

    /* Fix broken Material Icon avatars in chat messages */
    [data-testid="chatAvatarIcon-user"],
    [data-testid="chatAvatarIcon-assistant"] {
        font-size: 0 !important;
        overflow: hidden;
    }
    [data-testid="chatAvatarIcon-user"]::after {
        content: "🧑";
        font-size: 1.2rem;
    }
    [data-testid="chatAvatarIcon-assistant"]::after {
        content: "🤖";
        font-size: 1.2rem;
    }

    /* Hide ALL broken Material Symbol text globally (e.g. keyboard_double_arrow_up) */
    .material-symbols-rounded {
        font-size: 0 !important;
        line-height: 0 !important;
    }

    /* Streamlit scroll-to-bottom / back-to-top button */
    [data-testid="ScrollToBottomContainer"] button,
    [data-testid="stBottomBlockContainer"] button .material-symbols-rounded {
        font-size: 0 !important;
    }
    [data-testid="ScrollToBottomContainer"] button::after {
        content: "⬇";
        font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ───────────────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
        <h1>📈 Investment RAG Assistant</h1>
        <p>Your guide to Indian Small Savings & Investment schemes — powered by official documents</p>
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
        "This assistant helps you understand **Indian Investment Schemes** (PPF, NPS, SSY, etc.) using "
        "**Retrieval-Augmented Generation (RAG)**.\n\n"
        "Responses are grounded in official government circulars and guides."
    )
    st.divider()

# ── Avatar map (avoids broken Material-Icon font) ────────
AVATARS = {"user": "🧑", "assistant": "🤖"}

# ── Chat state ───────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS.get(msg["role"])):
        st.markdown(msg["content"], unsafe_allow_html=True)

# ── Chat input ───────────────────────────────────────────
if prompt := st.chat_input("Ask about PPF, Sukanya, NPS, or other schemes…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(prompt)

    # Get answer from backend
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            with st.spinner("Thinking…"):
                result = rag_engine.ask(prompt)

            # Build response with source badges
            answer_text = result["answer"]
            if result["sources"]:
                badges = " ".join(
                    f'<span class="source-badge">{s}</span>'
                    for s in result["sources"]
                )
                answer_text += f"\n\n📄 **Sources:** {badges}"

        except FileNotFoundError:
            logger.exception("Data directory not found")
            answer_text = (
                "⚠️ **Data folder not found.**\n\n"
                "Please make sure your investment documents (PDF/TXT) are in the `data/` folder "
                "and restart the app."
            )
        except Exception:
            logger.exception("Error processing query")
            answer_text = (
                "❌ **Something went wrong.**\n\n"
                "Please try again in a moment. If the issue persists, check the application logs."
            )

        st.markdown(answer_text, unsafe_allow_html=True)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer_text}
        )
