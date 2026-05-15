"""LBH Cima — Polished light theme chat UI."""

import base64
from pathlib import Path

import streamlit as st
from src.recommender import MovieRecommender


ROOT = Path(__file__).resolve().parent
LOGO_PATH = ROOT / "assets" / "logo.png"


# ────────────────────────────────────────────────────────────────────────
# Page config — must be the first Streamlit call
# ────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LBH Cima",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🎬",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ────────────────────────────────────────────────────────────────────────
# Custom CSS — ChatGPT-style light theme
# ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide Streamlit chrome */
    #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

    /* Global background and font */
    .stApp {
        background: #ffffff;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                     "Helvetica Neue", Arial, sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 7rem;
        max-width: 760px;
    }

    /* Sidebar — soft grey like ChatGPT */
    [data-testid="stSidebar"] {
        background: #f7f7f8;
        border-right: 1px solid #e5e5e5;
    }
    [data-testid="stSidebar"] * { color: #1a1a1a !important; }

    /* Sidebar buttons — pill style */
    [data-testid="stSidebar"] .stButton button {
        background: #ffffff !important;
        color: #1a1a1a !important;
        border: 1px solid #e5e5e5 !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        font-size: 0.875rem !important;
        text-align: left !important;
        font-weight: 400 !important;
        transition: all 0.15s ease;
        width: 100%;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: #1a1a1a !important;
        color: #ffffff !important;
        border-color: #1a1a1a !important;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.75rem 0 !important;
    }
    [data-testid="stChatMessageContent"] {
        font-size: 1rem;
        line-height: 1.65;
        color: #1a1a1a;
    }
    [data-testid="stChatMessageContent"] p {
        color: #1a1a1a !important;
    }

    /* User bubble — light grey */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
    [data-testid="stChatMessageContent"] {
        background: #f4f4f4;
        border-radius: 18px;
        padding: 12px 16px;
    }

    /* Chat input — sleek rounded bar */
    [data-testid="stChatInput"] {
        background: #ffffff !important;
        border: 1px solid #e5e5e5 !important;
        border-radius: 24px !important;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
    }
    [data-testid="stChatInput"] textarea {
        color: #1a1a1a !important;
        font-size: 1rem !important;
    }

    /* Title row — logo + name side by side */
    .title-row {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 0.25rem;
    }
    .title-row img {
        width: 48px;
        height: 48px;
        object-fit: contain;
    }
    .title-row h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #1a1a1a;
    }

    .subtitle {
        color: #6e6e80;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    hr { border-color: #e5e5e5; }

    /* Sidebar headings */
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────
# Cached recommender — load heavy assets once per session
# ────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_recommender():
    return MovieRecommender()


# ────────────────────────────────────────────────────────────────────────
# Header — logo + title
# ────────────────────────────────────────────────────────────────────────
if LOGO_PATH.exists():
    b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
    st.markdown(f"""
        <div class="title-row">
            <img src="data:image/png;base64,{b64}" alt="LBH Cima logo">
            <h1>LBH Cima</h1>
        </div>
    """, unsafe_allow_html=True)
else:
    st.title("🎬 LBH Cima")

st.markdown(
    '<div class="subtitle">'
    "AI movie recommender by Louai Ben Hassine. Tell me what you've watched, "
    "what mood you're in, or what you feel like tonight — "
    "I'll recommend from a database of 4,800 films."
    "</div>",
    unsafe_allow_html=True,
)


# ────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Try saying...")
    examples = [
        "I just watched Inception, what's similar?",
        "A dark psychological thriller from the 90s",
        "Feel-good comedy under 2 hours",
        "Something visually stunning like Blade Runner",
        "A movie to watch with my dad on a Sunday",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state.pending_input = ex

    st.divider()
    st.markdown("### How it works")
    st.markdown(
        "1. Your message becomes a vector embedding\n"
        "2. The 10 most semantically similar movies from a 4,800-film "
        "dataset are retrieved\n"
        "3. Gemini composes a response using only those real movies\n\n"
        "This is **Retrieval-Augmented Generation (RAG)**."
    )
    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()


# ────────────────────────────────────────────────────────────────────────
# Load the recommender (cached, runs once)
# ────────────────────────────────────────────────────────────────────────
with st.spinner("Waking up LBH Cima..."):
    rec = load_recommender()


# ────────────────────────────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None


# ────────────────────────────────────────────────────────────────────────
# Render chat history
# ────────────────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = str(LOGO_PATH) if (msg["role"] == "assistant" and LOGO_PATH.exists()) else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


# ────────────────────────────────────────────────────────────────────────
# Handle new input
# ────────────────────────────────────────────────────────────────────────
typed = st.chat_input("Tell me about a movie you liked, or what you're in the mood for...")
user_input = typed or st.session_state.pending_input
st.session_state.pending_input = None

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    avatar = str(LOGO_PATH) if LOGO_PATH.exists() else None
    with st.chat_message("assistant", avatar=avatar):
        with st.spinner("Picking movies for you..."):
            try:
                reply = rec.chat(user_input, st.session_state.history)
            except Exception as e:
                reply = (
                    f"⚠️ Something went wrong: `{e}`\n\n"
                    "If this keeps happening, you may have hit the Gemini "
                    "free-tier rate limit. Wait a minute and try again."
                )
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.history.append({"role": "user", "parts": [user_input]})
    st.session_state.history.append({"role": "model", "parts": [reply]})
