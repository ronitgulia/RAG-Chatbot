"""
Custom CSS — Iron Man aesthetic on pure white.
Orbitron (headings) + Rajdhani (body), red/gold accents, no sidebar.
"""
import streamlit as st

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');

/* ── Reset ─────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* Hide sidebar entirely */
[data-testid="stSidebar"], [data-testid="collapsedControl"] {
    display: none !important;
}

/* ── Base ───────────────────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
.main .block-container {
    background-color: #ffffff !important;
    color: #111111 !important;
    font-family: 'Rajdhani', sans-serif !important;
}

[data-testid="block-container"] {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Top Navigation Bar ─────────────────────────────────────────── */
.topbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 62px;
    background: #ffffff;
    border-bottom: 2px solid #e8e8e8;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 2rem;
    z-index: 9999;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

.topbar-brand {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.25rem;
    font-weight: 900;
    letter-spacing: 0.12em;
    color: #111111;
    text-transform: uppercase;
}

.topbar-brand span {
    color: #C0392B;
}

.topbar-right {
    display: flex;
    align-items: center;
    gap: 1.2rem;
}

.made-in-india {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #888;
    text-transform: uppercase;
    border: 1px solid #e0e0e0;
    padding: 3px 10px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    gap: 4px;
}

.india-flag {
    display: inline-flex;
    flex-direction: column;
    width: 14px;
    height: 10px;
    border-radius: 1px;
    overflow: hidden;
}

.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
}

.status-dot.connected { background: #27ae60; box-shadow: 0 0 6px #27ae6066; }
.status-dot.disconnected { background: #C0392B; box-shadow: 0 0 6px #C0392B66; }

/* ── Page Content Push (below fixed navbar) ─────────────────────── */
.page-wrapper {
    padding-top: 80px;
    padding-left: 2rem;
    padding-right: 2rem;
    padding-bottom: 2rem;
    max-width: 1000px;
    margin: 0 auto;
}

/* ── Headings ───────────────────────────────────────────────────── */
h1, h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    color: #111111 !important;
}

h1 { font-size: 1.6rem !important; }
h2 { font-size: 1.2rem !important; }
h3 { font-size: 1rem !important; }

.section-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #C0392B;
    margin-bottom: 1rem;
    padding-bottom: 6px;
    border-bottom: 1px solid #f0f0f0;
}

/* ── Buttons ────────────────────────────────────────────────────── */
.stButton > button {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    border: 1.5px solid #111111 !important;
    background-color: #ffffff !important;
    color: #111111 !important;
    padding: 0.45rem 1.2rem !important;
    transition: all 0.18s ease !important;
}

.stButton > button:hover {
    background-color: #C0392B !important;
    color: #ffffff !important;
    border-color: #C0392B !important;
    box-shadow: 0 4px 16px rgba(192,57,43,0.25) !important;
}

/* Primary button (Connect) */
.stButton > button[kind="primary"] {
    background-color: #C0392B !important;
    color: #ffffff !important;
    border-color: #C0392B !important;
}

.stButton > button[kind="primary"]:hover {
    background-color: #a93226 !important;
    box-shadow: 0 4px 16px rgba(192,57,43,0.35) !important;
}

/* ── Chat Messages ─────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background-color: #fafafa !important;
    border: 1px solid #ebebeb !important;
    border-radius: 6px !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: 0.75rem !important;
    font-family: 'Rajdhani', sans-serif !important;
}

[data-testid="stChatMessageContent"] p {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 400 !important;
    line-height: 1.65 !important;
    color: #222222 !important;
}

/* User message highlight */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background-color: #fff5f5 !important;
    border-color: #f5c6c6 !important;
    border-left: 3px solid #C0392B !important;
}

/* ── Chat Input ─────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    border: 1.5px solid #dddddd !important;
    border-radius: 6px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1rem !important;
    background-color: #ffffff !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #C0392B !important;
    box-shadow: 0 0 0 3px rgba(192,57,43,0.1) !important;
}

/* ── Dialog / Modal ─────────────────────────────────────────────── */
[data-testid="stModal"] {
    font-family: 'Rajdhani', sans-serif !important;
}

[data-testid="stModal"] h1,
[data-testid="stModal"] h2,
[data-testid="stModal"] h3 {
    font-family: 'Orbitron', sans-serif !important;
}

/* ── Tabs ───────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #888888 !important;
    padding: 0.5rem 1rem !important;
}

[data-testid="stTabs"] [aria-selected="true"] {
    color: #C0392B !important;
    border-bottom: 2px solid #C0392B !important;
}

/* ── Inputs ─────────────────────────────────────────────────────── */
input, textarea,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stPasswordInput"] input {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.95rem !important;
    border-radius: 4px !important;
    border: 1.5px solid #dddddd !important;
    color: #111111 !important;
    background-color: #ffffff !important;
}

input:focus, textarea:focus {
    border-color: #C0392B !important;
    box-shadow: 0 0 0 3px rgba(192,57,43,0.08) !important;
}

/* ── Selectbox ──────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    border-radius: 4px !important;
    border-color: #dddddd !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.95rem !important;
}

/* ── Metrics ─────────────────────────────────────────────────────  */
[data-testid="stMetric"] {
    background-color: #fafafa !important;
    border: 1px solid #ebebeb !important;
    border-radius: 6px !important;
    padding: 0.75rem 1rem !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 700 !important;
    color: #111111 !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #888888 !important;
}

/* ── Expander ───────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #ebebeb !important;
    border-radius: 6px !important;
    background-color: #fafafa !important;
}

[data-testid="stExpanderToggleIcon"] { color: #C0392B !important; }

/* ── File Uploader ──────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 2px dashed #dddddd !important;
    border-radius: 6px !important;
    background-color: #fafafa !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: #C0392B !important;
}

/* ── Chunk card ─────────────────────────────────────────────────── */
.chunk-card {
    background: #fafafa;
    border: 1px solid #ebebeb;
    border-left: 3px solid #C0392B;
    border-radius: 4px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
    font-family: 'Rajdhani', sans-serif;
}

.chunk-source {
    font-size: 0.75rem;
    font-weight: 700;
    color: #C0392B;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    display: block;
    margin-bottom: 0.3rem;
}

.chunk-text {
    font-size: 0.88rem;
    color: #444;
    line-height: 1.5;
}

/* ── Alerts ─────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 4px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}

/* ── Dataframe ──────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #ebebeb !important;
    border-radius: 6px !important;
}

/* ── Divider ─────────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #ebebeb !important;
    margin: 1.25rem 0 !important;
}

/* ── Radio buttons ──────────────────────────────────────────────── */
[data-testid="stRadio"] label {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
}

/* ── Slider ─────────────────────────────────────────────────────── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background-color: #C0392B !important;
}

/* ── Toggle ─────────────────────────────────────────────────────── */
[data-testid="stToggle"] [aria-checked="true"] {
    background-color: #C0392B !important;
}

/* ── Caption / small text ───────────────────────────────────────── */
[data-testid="stCaptionContainer"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.8rem !important;
    color: #999 !important;
}

/* ── Scrollbar ──────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #f5f5f5; }
::-webkit-scrollbar-thumb { background: #ddd; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #C0392B; }

/* ── Footer strip ───────────────────────────────────────────────── */
.footer-strip {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #bbbbbb;
    text-align: center;
    padding: 1.5rem 0 0.5rem;
}

.footer-strip span { color: #C0392B; }

/* Hide Streamlit default footer & hamburger */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }

/* Spinner text */
[data-testid="stSpinner"] p {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    color: #888 !important;
}
"""


def inject_css():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


inject_css()