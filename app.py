"""LBH Cima — glassy light-mode chat UI (Powered by Local Ollama)."""

import base64
from datetime import datetime
from pathlib import Path

import streamlit as st
from src.recommender import MovieRecommender


ROOT = Path(__file__).resolve().parent
LOGO_PATH = ROOT / "assets" / "logo.png"

# Name shown in the hero greeting — change this to whatever you want.
USER_NAME = "Louai"


# ────────────────────────────────────────────────────────────────────────
# Page config — must be the first Streamlit call
# ────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LBH Cima",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def greeting() -> str:
    """Good Morning / Afternoon / Evening based on the local clock."""
    hour = datetime.now().hour
    if hour < 12:
        return "Good Morning"
    if hour < 18:
        return "Good Afternoon"
    return "Good Evening"


# Writing styles are appended to the prompt, so the RAG backend is untouched.
STYLE_HINTS = {
    "Default": "",
    "Concise": "Answer in a compact way: one short sentence per film, no preamble.",
    "Detailed": "Go deeper on tone, cinematography and exactly why each film fits.",
    "Playful": "Keep the tone witty and playful, but still accurate.",
}

EXAMPLES = [
    "I just watched Inception, what's similar?",
    "A dark psychological thriller from the 90s",
    "Feel-good comedy under 2 hours",
    "Something visually stunning like Blade Runner",
]


# ────────────────────────────────────────────────────────────────────────
# Custom CSS
#
# Note on the dark bottom strip: Streamlit renders the chat input inside
# [data-testid="stBottom"], which holds an unnamed emotion-styled wrapper
# div that paints its own background from the *active* theme. Overriding
# .stApp alone never reaches it, which is why the input appeared to float
# on a black bar. The rules below repaint that whole subtree, and
# .streamlit/config.toml pins the theme to light so the dark colours are
# never generated in the first place.
# ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Force the browser itself into light rendering ─────────────────
       Without this, form controls, scrollbars and the textarea caret are
       still painted with the OS dark-mode widget colours. */
    :root, html, body, .stApp {
        color-scheme: light only !important;
    }

    :root {
        --background-color: #ffffff;
        --secondary-background-color: #f7f7f8;
        --text-color: #1a1a1a;
        --primary-color: #a855f7;
        --rail-w: 64px;
    }

    /* ── Typography — system stack, so the app stays fully offline ─────*/
    html, body, .stApp, [data-testid="stChatMessageContent"],
    [data-testid="stChatInput"] textarea, textarea, input, button {
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI",
                     "Helvetica Neue", Arial, sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    /* Hide Streamlit chrome + the unused sidebar */
    #MainMenu, footer,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }

    /* ── Global light surfaces ─────────────────────────────────────────*/
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    [data-testid="stHeader"] {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
    }
    [data-testid="stHeader"] { box-shadow: none !important; height: 0 !important; }

    /* Leave room for the fixed icon rail */
    [data-testid="stAppViewContainer"] { padding-left: var(--rail-w); }

    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 7rem;
        max-width: 820px;
    }

    /* ══════════════════════════════════════════════════════════════════
       Bottom / footer wrapper forced to pure white.
       Every layer Streamlit stacks under the chat input is repainted.
       The chat input is a separate testid, so it keeps its border+glow.
       ══════════════════════════════════════════════════════════════════ */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottom"] > div > div,
    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottom"] [data-testid="stVerticalBlock"],
    [data-testid="stBottom"] [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stBottom"] [data-testid="stElementContainer"],
    [data-testid="stChatInputContainer"] {
        background: #ffffff !important;
        background-color: #ffffff !important;
        border: none !important;
        border-top: none !important;
        box-shadow: none !important;
    }
    [data-testid="stBottom"]::before,
    [data-testid="stBottom"]::after {
        display: none !important;
        background: #ffffff !important;
    }
    [data-testid="stBottom"] { padding-left: var(--rail-w) !important; }
    [data-testid="stBottomBlockContainer"] {
        max-width: 820px;
        padding-bottom: 1.5rem !important;
    }

    /* ══════════════════════════════════════════════════════════════════
       LEFT ICON RAIL (fixed, decorative)
       ══════════════════════════════════════════════════════════════════ */
    .rail {
        position: fixed;
        top: 0; left: 0; bottom: 0;
        width: var(--rail-w);
        background: #ffffff;
        border-right: 1px solid #ececf1;
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 14px 0 16px;
        z-index: 999;
    }
    .rail .spacer { flex: 1; }
    .rail-logo {
        width: 30px; height: 30px;
        border-radius: 9px;
        object-fit: contain;
        margin-bottom: 16px;
    }
    .rail-btn {
        width: 34px; height: 34px;
        display: flex; align-items: center; justify-content: center;
        border-radius: 9px;
        margin-bottom: 5px;
        color: #8e8ea0;
        transition: background .15s ease, color .15s ease;
    }
    .rail-btn svg { width: 17px; height: 17px; }
    .rail-btn:hover { background: #f2f2f5; color: #1a1a1a; }
    .rail-btn.active { background: #f4ecff; color: #a855f7; }
    .rail-avatar {
        width: 28px; height: 28px;
        border-radius: 50%;
        object-fit: cover;
        border: 1px solid #e5e5e5;
    }

    /* ══════════════════════════════════════════════════════════════════
       TOP BAR
       ══════════════════════════════════════════════════════════════════ */
    .st-key-topbar [data-testid="stHorizontalBlock"] { align-items: center; }
    .model-pill {
        display: inline-flex; align-items: center; gap: 7px;
        border: 1px solid #e5e5e5;
        border-radius: 10px;
        padding: 6px 11px;
        font-size: 0.82rem;
        font-weight: 500;
        color: #1a1a1a;
        background: #ffffff;
        white-space: nowrap;
    }
    .model-pill .dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: linear-gradient(135deg, #d946ef, #a855f7);
        box-shadow: 0 0 7px rgba(217, 70, 239, .8);
    }
    .model-pill .chev { color: #9ca3af; font-size: 0.7rem; }

    /* "New Thread" — solid dark pill, right aligned */
    .st-key-topbar .stButton { display: flex; justify-content: flex-end; }
    .st-key-topbar .stButton button {
        background: #17171a !important;
        color: #ffffff !important;
        border: 1px solid #17171a !important;
        border-radius: 10px !important;
        padding: 7px 15px !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        box-shadow: none !important;
        transition: transform .12s ease, background .15s ease;
    }
    .st-key-topbar .stButton button:hover {
        background: #000000 !important;
        transform: translateY(-1px);
    }
    .st-key-topbar .stButton button * { color: #ffffff !important; }

    /* ══════════════════════════════════════════════════════════════════
       HERO — the glowing orb
       ══════════════════════════════════════════════════════════════════ */
    .hero { text-align: center; padding: 42px 0 26px; }

    .orb {
        position: relative;
        width: 74px; height: 74px;
        margin: 0 auto 30px;
        border-radius: 50%;
        background:
            radial-gradient(circle at 31% 27%,
                #ffffff 0%,
                #fbe4ff 9%,
                #efaef8 26%,
                #d872ef 48%,
                #b445dd 70%,
                #8f2fc0 100%);
        box-shadow:
            inset 0 0 0 1px rgba(255,255,255,.45),
            inset -6px -8px 18px rgba(108, 24, 150, .35),
            0 12px 30px rgba(184, 70, 221, .42),
            0 0 44px 8px rgba(214, 108, 244, .42),
            0 0 90px 26px rgba(214, 108, 244, .22);
        animation: orbFloat 6s ease-in-out infinite;
    }
    /* soft bloom behind the sphere */
    .orb::before {
        content: "";
        position: absolute;
        inset: -42px;
        border-radius: 50%;
        background: radial-gradient(circle,
            rgba(214,108,244,.30) 0%,
            rgba(214,108,244,.12) 40%,
            rgba(214,108,244,0) 70%);
        filter: blur(8px);
        z-index: -1;
        animation: orbPulse 5s ease-in-out infinite;
    }
    /* specular highlight */
    .orb::after {
        content: "";
        position: absolute;
        top: 11%; left: 20%;
        width: 30%; height: 22%;
        border-radius: 50%;
        background: radial-gradient(ellipse at center,
            rgba(255,255,255,.95) 0%, rgba(255,255,255,0) 72%);
        transform: rotate(-18deg);
    }
    @keyframes orbFloat {
        0%, 100% { transform: translateY(0); }
        50%      { transform: translateY(-7px); }
    }
    @keyframes orbPulse {
        0%, 100% { opacity: .85; transform: scale(1); }
        50%      { opacity: 1;   transform: scale(1.06); }
    }

    .hero h1 {
        margin: 0;
        font-size: 2.35rem;
        line-height: 1.28;
        font-weight: 600;
        letter-spacing: -0.025em;
        color: #16161a;
    }
    .hero h1 .accent {
        background: linear-gradient(92deg, #a855f7 0%, #d946ef 55%, #e879f9 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        color: transparent;
    }

    /* ══════════════════════════════════════════════════════════════════
       COMPOSER CARD (the st.form on the landing screen)
       ══════════════════════════════════════════════════════════════════ */
    [data-testid="stForm"] {
        position: relative;
        border: 1px solid #e6e6ea !important;
        border-radius: 18px !important;
        background: #ffffff !important;
        padding: 16px 16px 12px !important;
        box-shadow: 0 6px 24px rgba(16, 16, 20, .07),
                    0 1px 3px rgba(16, 16, 20, .05) !important;
        transition: border-color .18s ease, box-shadow .18s ease;
    }
    [data-testid="stForm"]:focus-within {
        border-color: #d8b4fe !important;
        box-shadow: 0 0 0 4px rgba(168, 85, 247, .10),
                    0 8px 30px rgba(168, 85, 247, .14) !important;
    }
    /* sparkle glyph to the left of the placeholder */
    [data-testid="stForm"]::before {
        content: "";
        position: absolute;
        top: 26px; left: 26px;
        width: 15px; height: 15px;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%23a1a1b5'%3E%3Cpath d='M12 2l1.6 5.6L19 9.2l-5.4 1.6L12 16l-1.6-5.2L5 9.2l5.4-1.6z'/%3E%3Cpath d='M18.5 14l.8 2.7L22 17.5l-2.7.8-.8 2.7-.8-2.7-2.7-.8 2.7-.8z'/%3E%3C/svg%3E");
        background-size: contain;
        background-repeat: no-repeat;
        pointer-events: none;
        z-index: 2;
    }
    [data-testid="stForm"] textarea {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 4px 4px 4px 26px !important;
        font-size: 0.98rem !important;
        color: #1a1a1a !important;
        resize: none !important;
        -webkit-text-fill-color: #1a1a1a !important;
    }
    [data-testid="stForm"] textarea::placeholder {
        color: #a1a1b5 !important;
        -webkit-text-fill-color: #a1a1b5 !important;
        opacity: 1 !important;
    }
    [data-testid="stForm"] [data-baseweb="textarea"],
    [data-testid="stForm"] [data-baseweb="base-input"] {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    /* hide the "Press Ctrl+Enter to submit" helper Streamlit adds */
    [data-testid="stForm"] [data-testid="InputInstructions"] { display: none !important; }
    [data-testid="stForm"] [data-testid="stHorizontalBlock"] { align-items: center; }

    /* Attach pill (decorative) */
    .attach-pill {
        display: inline-flex; align-items: center;
        border: 1px solid #e6e6ea;
        border-radius: 999px;
        padding: 7px 14px;
        font-size: 0.8rem;
        color: #45454f;
        background: #ffffff;
        white-space: nowrap;
        user-select: none;
        line-height: 1;
    }
    .attach-pill svg {
        width: 13px; height: 13px;
        margin-right: 7px;
        flex: 0 0 auto;
        stroke: #6e6e80;
    }

    /* Writing Styles — selectbox styled as a pill */
    [data-testid="stForm"] [data-testid="stSelectbox"] > div > div {
        border: 1px solid #e6e6ea !important;
        border-radius: 999px !important;
        background: #ffffff !important;
        min-height: 33px !important;
        font-size: 0.8rem !important;
        box-shadow: none !important;
    }
    [data-testid="stForm"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        padding-top: 2px !important;
        padding-bottom: 2px !important;
    }

    /* Citation toggle.
       BaseWeb paints the track on the label's FIRST child div, and it already
       picks up the purple primaryColor from .streamlit/config.toml — so only
       the text needs sizing here. (Targeting `input:checked + div` would hit
       the label-text wrapper instead and highlight the word "Citation".) */
    [data-testid="stForm"] [data-baseweb="checkbox"] [data-testid="stWidgetLabel"] p {
        font-size: .8rem !important;
        color: #45454f !important;
    }

    /* Send button — dark circle with the up arrow */
    [data-testid="stFormSubmitButton"] { display: flex; justify-content: flex-end; }
    [data-testid="stFormSubmitButton"] button {
        background: #17171a !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 50% !important;
        width: 34px !important; height: 34px !important;
        min-height: 34px !important;
        padding: 0 !important;
        font-size: 1rem !important;
        line-height: 1 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,.18) !important;
        transition: transform .12s ease, background .15s ease;
    }
    [data-testid="stFormSubmitButton"] button:hover {
        background: #000 !important;
        transform: translateY(-1px);
    }
    [data-testid="stFormSubmitButton"] button * { color: #ffffff !important; }

    /* ══════════════════════════════════════════════════════════════════
       EXAMPLE CARDS
       ══════════════════════════════════════════════════════════════════ */
    .examples-label {
        margin: 34px 0 12px;
        font-size: 0.68rem;
        letter-spacing: .09em;
        text-transform: uppercase;
        color: #9a9aa8;
        font-weight: 500;
    }
    .st-key-examples [data-testid="stColumn"] { padding: 0 5px; }
    .st-key-examples .stButton button {
        position: relative;
        background: #f7f7f8 !important;
        color: #45454f !important;
        border: 1px solid #f0f0f3 !important;
        border-radius: 14px !important;
        padding: 15px 14px 42px !important;
        height: 122px !important;   /* fixed, so a 3-line card can't out-grow a 2-line one */
        font-size: 0.83rem !important;
        font-weight: 400 !important;
        line-height: 1.42 !important;
        text-align: left !important;
        align-items: flex-start !important;
        justify-content: flex-start !important;
        white-space: normal !important;
        width: 100%;
        box-shadow: none !important;
        transition: background .15s ease, transform .12s ease, box-shadow .15s ease;
    }
    .st-key-examples .stButton button:hover {
        background: #f1f1f4 !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(16,16,20,.07) !important;
        border-color: #e8e8ee !important;
    }
    .st-key-examples .stButton button p {
        text-align: left !important;
        color: #45454f !important;
        margin: 0 !important;
    }
    /* per-card line icon, pinned bottom-left */
    .st-key-examples .stButton button::after {
        content: "";
        position: absolute;
        left: 15px; bottom: 14px;
        width: 17px; height: 17px;
        background-repeat: no-repeat;
        background-size: contain;
        opacity: .75;
    }
    .st-key-examples [data-testid="stColumn"]:nth-child(1) .stButton button::after {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%236e6e80' stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='2' y='3' width='20' height='18' rx='2'/%3E%3Cline x1='7' y1='3' x2='7' y2='21'/%3E%3Cline x1='17' y1='3' x2='17' y2='21'/%3E%3C/svg%3E");
    }
    .st-key-examples [data-testid="stColumn"]:nth-child(2) .stButton button::after {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%236e6e80' stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z'/%3E%3C/svg%3E");
    }
    .st-key-examples [data-testid="stColumn"]:nth-child(3) .stButton button::after {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%236e6e80' stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpath d='M8 14s1.5 2 4 2 4-2 4-2'/%3E%3Cline x1='9' y1='9' x2='9.01' y2='9'/%3E%3Cline x1='15' y1='9' x2='15.01' y2='9'/%3E%3C/svg%3E");
    }
    .st-key-examples [data-testid="stColumn"]:nth-child(4) .stButton button::after {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%236e6e80' stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z'/%3E%3Ccircle cx='12' cy='12' r='3'/%3E%3C/svg%3E");
    }

    /* ══════════════════════════════════════════════════════════════════
       CHAT
       ══════════════════════════════════════════════════════════════════ */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.75rem 0 !important;
    }
    [data-testid="stChatMessageContent"] {
        font-size: 1rem;
        line-height: 1.65;
        color: #1a1a1a !important;
    }
    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] li,
    [data-testid="stChatMessageContent"] strong { color: #1a1a1a !important; }

    /* User bubble.
       Streamlit renamed this testid: 1.39+ uses stChatMessageAvatarUser,
       older builds used chatAvatarIcon-user. Both are matched so the grey
       bubble survives a version bump in either direction. */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
    [data-testid="stChatMessageContent"] {
        background: #f4f4f4 !important;
        border-radius: 18px;
        padding: 12px 16px;
    }

    /* Chat input (chat view) */
    [data-testid="stChatInput"] {
        background: #ffffff !important;
        border: 1px solid #e6e6ea !important;
        border-radius: 18px !important;
        box-shadow: 0 6px 24px rgba(16,16,20,.08) !important;
    }
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div {
        background: transparent !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #d8b4fe !important;
        box-shadow: 0 0 0 4px rgba(168,85,247,.10),
                    0 8px 30px rgba(168,85,247,.14) !important;
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInputTextArea"] {
        background: transparent !important;
        color: #1a1a1a !important;
        font-size: 1rem !important;
        caret-color: #1a1a1a !important;
        -webkit-text-fill-color: #1a1a1a !important;
    }
    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInputTextArea"]::placeholder {
        color: #a1a1b5 !important;
        -webkit-text-fill-color: #a1a1b5 !important;
        opacity: 1 !important;
    }
    [data-testid="stChatInput"] button,
    [data-testid="stChatInputSubmitButton"] {
        background: #17171a !important;
        border-radius: 50% !important;
        border: none !important;
        width: 32px !important; height: 32px !important;
    }
    [data-testid="stChatInput"] button svg,
    [data-testid="stChatInputSubmitButton"] svg {
        fill: #ffffff !important; color: #ffffff !important;
    }

    /* Sources expander */
    [data-testid="stExpander"] {
        border: 1px solid #eeeef2 !important;
        border-radius: 12px !important;
        background: #fbfbfc !important;
    }
    [data-testid="stExpander"] summary { font-size: .82rem !important; color: #6e6e80 !important; }

    [data-testid="stSpinner"] { color: #8e8ea0 !important; }
    hr { border-color: #ececf1 !important; }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #ffffff; }
    ::-webkit-scrollbar-thumb { background: #dcdce4; border-radius: 8px; }
    ::-webkit-scrollbar-thumb:hover { background: #c2c2ce; }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────
# Fixed left icon rail (decorative)
# ────────────────────────────────────────────────────────────────────────
def _svg(path: str) -> str:
    return (
        "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' "
        f"stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'>{path}</svg>"
    )


RAIL_ICONS = [
    ("home", "<path d='M3 9.5L12 3l9 6.5V20a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z'/>"),
    ("chat", "<path d='M21 11.5a8.4 8.4 0 0 1-9 8.4 8.9 8.9 0 0 1-4-.9L3 21l1.9-4.6A8.4 8.4 0 0 1 12 3a8.4 8.4 0 0 1 9 8.5z'/>"),
    ("clock", "<circle cx='12' cy='12' r='9'/><polyline points='12 7 12 12 15 14'/>"),
    ("bolt", "<polygon points='13 2 4 14 11 14 10 22 20 10 13 10 13 2'/>"),
    ("card", "<rect x='2' y='5' width='20' height='14' rx='2'/><line x1='2' y1='10' x2='22' y2='10'/>"),
    ("share", "<circle cx='18' cy='5' r='3'/><circle cx='6' cy='12' r='3'/><circle cx='18' cy='19' r='3'/><line x1='8.6' y1='10.6' x2='15.4' y2='6.4'/><line x1='8.6' y1='13.4' x2='15.4' y2='17.6'/>"),
    ("db", "<ellipse cx='12' cy='6' rx='8' ry='3'/><path d='M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6'/><path d='M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6'/>"),
]

logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode() if LOGO_PATH.exists() else ""
logo_tag = (
    f"<img class='rail-logo' src='data:image/png;base64,{logo_b64}' alt='LBH Cima'>"
    if logo_b64 else "<div class='rail-logo'></div>"
)
avatar_tag = (
    f"<img class='rail-avatar' src='data:image/png;base64,{logo_b64}' alt='profile'>"
    if logo_b64 else ""
)

rail_buttons = "".join(
    f"<div class='rail-btn{' active' if name == 'home' else ''}' title='{name}'>{_svg(path)}</div>"
    for name, path in RAIL_ICONS
)

st.markdown(
    f"""
    <div class="rail">
        {logo_tag}
        {rail_buttons}
        <div class="spacer"></div>
        <div class="rail-btn" title="support">
            {_svg("<path d='M3 18v-6a9 9 0 0 1 18 0v6'/><path d='M21 19a2 2 0 0 1-2 2h-1a2 2 0 0 1-2-2v-3a2 2 0 0 1 2-2h3zM3 19a2 2 0 0 0 2 2h1a2 2 0 0 0 2-2v-3a2 2 0 0 0-2-2H3z'/>")}
        </div>
        <div class="rail-btn" title="settings">
            {_svg("<circle cx='12' cy='12' r='3'/><path d='M19.4 15a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-1.8-.3 1.6 1.6 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1A1.6 1.6 0 0 0 9 19.4a1.6 1.6 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.6 1.6 0 0 0 .3-1.8 1.6 1.6 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1A1.6 1.6 0 0 0 4.6 9a1.6 1.6 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 1.8.3H9a1.6 1.6 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 1 1.5 1.6 1.6 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0-.3 1.8V9a1.6 1.6 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1z'/>")}
        </div>
        {avatar_tag}
    </div>
    """,
    unsafe_allow_html=True,
)


# ────────────────────────────────────────────────────────────────────────
# Cached recommender — load heavy assets once per session
# ────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_recommender():
    return MovieRecommender()


# ────────────────────────────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None
if "style" not in st.session_state:
    st.session_state.style = "Default"
if "citation" not in st.session_state:
    st.session_state.citation = True


# ────────────────────────────────────────────────────────────────────────
# Top bar
# ────────────────────────────────────────────────────────────────────────
with st.container(key="topbar"):
    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            "<div class='model-pill'><span class='dot'></span>"
            "llama3.2 · local <span class='chev'>▾</span></div>",
            unsafe_allow_html=True,
        )
    with right:
        if st.button("＋  New Thread", key="new_thread"):
            st.session_state.messages = []
            st.session_state.history = []
            st.session_state.pending_input = None
            st.rerun()


# ────────────────────────────────────────────────────────────────────────
# Load the recommender (cached, runs once)
# ────────────────────────────────────────────────────────────────────────
with st.spinner("Waking up local Ollama engine..."):
    rec = load_recommender()


show_hero = not st.session_state.messages and not st.session_state.pending_input


# ────────────────────────────────────────────────────────────────────────
# Landing screen — orb, greeting, composer, examples
# ────────────────────────────────────────────────────────────────────────
if show_hero:
    st.markdown(
        f"""
        <div class="hero">
            <div class="orb"></div>
            <h1>{greeting()}, {USER_NAME}<br>
                What's on <span class="accent">your mind?</span></h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("composer", clear_on_submit=True, border=True):
        prompt = st.text_area(
            "Prompt",
            placeholder="Ask AI a question or make a request.",
            label_visibility="collapsed",
            height=78,
            key="composer_text",
        )

        c1, c2, c3, c4, c5 = st.columns([1.05, 1.5, 1.7, 1.15, 0.55],
                                        vertical_alignment="center")
        with c1:
            clip = _svg(
                "<path d='M21.4 11.05l-9.2 9.2a5 5 0 0 1-7.1-7.1l9.2-9.2a3.3 3.3 0 1 1 4.7 4.7"
                "l-9.2 9.2a1.7 1.7 0 0 1-2.4-2.4l8.5-8.5'/>"
            )
            st.markdown(
                f"<div class='attach-pill'>{clip}Attach</div>", unsafe_allow_html=True
            )
        with c2:
            # index=None keeps "Writing Styles" showing as the pill label,
            # exactly like the reference, until a style is actually picked.
            style = st.selectbox(
                "Writing Styles",
                list(STYLE_HINTS.keys()),
                index=None,
                placeholder="Writing Styles",
                label_visibility="collapsed",
                key="style_select",
            )
        with c3:
            st.write("")
        with c4:
            citation = st.toggle("Citation", value=st.session_state.citation)
        with c5:
            sent = st.form_submit_button("↑")

    # Handled outside the `with st.form(...)` block: calling st.rerun() while
    # still inside the form context does not reliably re-enter the script.
    if sent and prompt and prompt.strip():
        st.session_state.style = style or "Default"
        st.session_state.citation = citation
        st.session_state.pending_input = prompt.strip()
        st.rerun()

    st.markdown(
        "<div class='examples-label'>Get started with an example below</div>",
        unsafe_allow_html=True,
    )
    with st.container(key="examples"):
        cols = st.columns(4)
        for col, ex in zip(cols, EXAMPLES):
            with col:
                if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                    st.session_state.pending_input = ex
                    st.rerun()


# ────────────────────────────────────────────────────────────────────────
# Chat view
# ────────────────────────────────────────────────────────────────────────
else:
    for msg in st.session_state.messages:
        avatar = str(LOGO_PATH) if (msg["role"] == "assistant" and LOGO_PATH.exists()) else None
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Citations — retrieved from the 4,800-film dataset"):
                    st.markdown(msg["sources"])


typed = st.chat_input("Ask AI a question or make a request.") if not show_hero else None
user_input = typed or st.session_state.pending_input
st.session_state.pending_input = None


# ────────────────────────────────────────────────────────────────────────
# Handle new input with streaming typewriter effect
# ────────────────────────────────────────────────────────────────────────
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Writing style is appended to the prompt, so the RAG backend is untouched.
    hint = STYLE_HINTS.get(st.session_state.style, "")
    prompt_for_model = f"{user_input}\n\n({hint})" if hint else user_input

    sources_md = ""
    if st.session_state.citation:
        try:
            hits = rec.retrieve(user_input, top_k=10)
            sources_md = "\n".join(
                f"- **{row['title']}** — similarity `{row['similarity']:.3f}`"
                for _, row in hits.iterrows()
            )
        except Exception:
            sources_md = ""

    avatar = str(LOGO_PATH) if LOGO_PATH.exists() else None
    with st.chat_message("assistant", avatar=avatar):
        message_placeholder = st.empty()
        full_response = ""
        try:
            # Stream response word-by-word from local Ollama model
            for chunk in rec.chat_stream(prompt_for_model, st.session_state.history):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            full_response = (
                f"⚠️ Ollama error: `{e}`\n\n"
                "Make sure the Ollama desktop application is actively running on your Mac!"
            )
            message_placeholder.markdown(full_response)

        if sources_md:
            with st.expander("Citations — retrieved from the 4,800-film dataset"):
                st.markdown(sources_md)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response, "sources": sources_md}
    )
    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "assistant", "content": full_response})

    if typed is None:
        st.rerun()
