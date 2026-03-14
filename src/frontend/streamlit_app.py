"""
Knowledge RAG Assistant — Streamlit Chat UI.

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
    page_title="Knowledge RAG Assistant",
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

    /* Header gradient - Knowledge theme */
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
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
        background: #334155;
        color: #e2e8f0;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 999px;
        margin: 0 3px;
        border: 1px solid #475569;
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
        <h1>📈 Knowledge RAG Assistant</h1>
        <p>Chat with your documents</p>
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
        "This assistant helps you understand your documents using "
        "**Retrieval-Augmented Generation (RAG)**.\n\n"
        "Responses are grounded purely in the provided content."
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


def handle_stream(stream):
    """Consume the stream, handle the sources metadata yield, and return the final strings."""
    full_answer = ""
    sources_str = ""
    
    for chunk in stream:
        # Check if this chunk is the special metadata chunk
        if chunk.startswith("\n\n[SOURCES_METADATA:"):
            sources_str = chunk.replace("\n\n[SOURCES_METADATA:", "").replace("]", "")
        elif chunk.startswith("🔑 **API Key") or chunk.startswith("🌐 **Connection ") or chunk.startswith("⚠️ **Data Not Found") or chunk.startswith("❌ **An unexpected error "):
            full_answer += chunk
            yield chunk
        else:
            # Yield exactly what came from the pipeline for smooth streaming
            full_answer += chunk
            yield chunk
            
    # We yield the sources at the VERY end as a normal markdown string if available
    # However st.write_stream yields the chunks sequentially. 
    # To append badges we must do it after write_stream finishes, so we store the state in st.session_state temporarily
    st.session_state.temp_sources = sources_str
    st.session_state.temp_full_answer = full_answer


# ── Chat input ───────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your documents…"):
    # We copy the history up to this point because ask_stream needs it
    chat_history = list(st.session_state.messages)
    
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(prompt)

    # Get answer from backend
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            # Show a status indicator while the first chunk is being prepared (RAG, searching, etc.)
            status_container = st.empty()
            with status_container.status("🔍 Searching knowledge base...", expanded=False) as status:
                stream = rag_engine.ask_stream(prompt, chat_history=chat_history)
                
                # We wrap the handle_stream in another generator that clears the status 
                # as soon as the first real text chunk arrives
                def stream_with_status_management(s):
                    first_text_chunk = True
                    for chunk in s:
                        # Once we get a real chunk (not the metadata one), clear the spinner
                        if first_text_chunk and not chunk.startswith("\n\n[SOURCES_METADATA:"):
                            status.update(label="✅ Thinking complete!", state="complete", expanded=False)
                            status_container.empty()
                            first_text_chunk = False
                        yield chunk

                # Use write_stream to display the text as it is generated
                st.write_stream(handle_stream(stream_with_status_management(stream)))
            
            final_answer = st.session_state.get("temp_full_answer", "")
            sources_str = st.session_state.get("temp_sources", "")
            
            # If sources exist, append them to the UI as markdown (which write_stream can't do natively for HTML spans easily)
            if sources_str:
                source_list = sources_str.split(",")
                badges = " ".join(
                    f'<span class="source-badge">{s}</span>'
                    for s in source_list
                )
                sources_md = f"\n\n📄 **Sources:** {badges}"
                # Append to screen immediately
                st.markdown(sources_md, unsafe_allow_html=True)
                final_answer += sources_md
            
            # Save the complete final answer (with sources) to the history
            st.session_state.messages.append(
                {"role": "assistant", "content": final_answer}
            )
            
        except Exception as e:
            logger.exception("Frontend error")
            answer_text = (
                "❌ **Something went wrong.**\n\n"
                f"Error: `{str(e)}`"
            )
            st.error(answer_text)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer_text}
            )
