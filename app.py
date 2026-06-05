"""
PitWall Intel — Main Streamlit Application
F1 Race Result Predictor with full carbon-dark pit wall aesthetic.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import sys
import requests
import time
from datetime import datetime

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
UTILS_DIR = os.path.join(BASE_DIR, "utils")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PitWall Intel",
    page_icon="assets/f1_logo.png" if os.path.exists("assets/f1_logo.png") else "🏎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# FULL CUSTOM CSS — Carbon Fibre Dark + F1 Brand
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;600;700;800&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');

/* ── ROOT VARIABLES ─────────────────────────────── */
:root {
    --f1-red:       #e8002d;
    --f1-yellow:    #ffcd00;
    --f1-white:     #f5f5f5;
    --carbon-bg:    #080808;
    --carbon-mid:   #111111;
    --carbon-card:  #161616;
    --carbon-border:#222222;
    --carbon-line:  #2a2a2a;
    --glow-red:     0 0 12px rgba(232,0,45,0.5);
    --glow-yellow:  0 0 12px rgba(255,205,0,0.4);
    --font-display: 'Barlow Condensed', sans-serif;
    --font-mono:    'Share Tech Mono', monospace;
    --font-ui:      'Rajdhani', sans-serif;
}

/* ── GLOBAL ─────────────────────────────────────── */
* { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: var(--font-ui) !important;
    background-color: var(--carbon-bg) !important;
    color: var(--f1-white) !important;
}
.stApp { background-color: var(--carbon-bg) !important; }

/* ── CARBON FIBRE TEXTURE ───────────────────────── */
.stApp::before {
    content: '';
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        repeating-linear-gradient(
            45deg,
            transparent,
            transparent 2px,
            rgba(255,255,255,0.012) 2px,
            rgba(255,255,255,0.012) 4px
        ),
        repeating-linear-gradient(
            -45deg,
            transparent,
            transparent 2px,
            rgba(255,255,255,0.008) 2px,
            rgba(255,255,255,0.008) 4px
        );
    pointer-events: none;
    z-index: 0;
}

/* ── SIDEBAR ────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d0d 0%, #0a0a0a 100%) !important;
    border-right: 1px solid var(--carbon-border) !important;
}
[data-testid="stSidebar"]::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--f1-red), var(--f1-yellow));
}
[data-testid="stSidebar"] * {
    color: var(--f1-white) !important;
    font-family: var(--font-ui) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: rgba(245,245,245,0.6) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

/* ── SELECTBOXES ────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background: var(--carbon-card) !important;
    border: 1px solid var(--carbon-border) !important;
    color: var(--f1-white) !important;
    font-family: var(--font-ui) !important;
    border-radius: 2px !important;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: var(--f1-red) !important;
}

/* ── BUTTONS ────────────────────────────────────── */
.stButton > button {
    background: var(--f1-red) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #ff1a3e !important;
    box-shadow: var(--glow-red) !important;
    transform: translateY(-1px) !important;
}

/* ── METRICS ────────────────────────────────────── */
[data-testid="metric-container"] {
    background: var(--carbon-card) !important;
    border: 1px solid var(--carbon-border) !important;
    border-top: 2px solid var(--f1-red) !important;
    border-radius: 2px !important;
    padding: 1rem !important;
}
[data-testid="metric-container"] label {
    font-family: var(--font-mono) !important;
    font-size: 0.65rem !important;
    color: rgba(245,245,245,0.5) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: var(--f1-white) !important;
}

/* ── DATAFRAMES ─────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--carbon-border) !important;
    border-radius: 2px !important;
}

/* ── TABS ───────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--carbon-mid) !important;
    border-bottom: 1px solid var(--carbon-border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--font-display) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: rgba(245,245,245,0.5) !important;
    border-radius: 0 !important;
    padding: 0.7rem 1.5rem !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: var(--f1-white) !important;
    border-bottom: 2px solid var(--f1-red) !important;
}

/* ── PROGRESS / SPINNER ─────────────────────────── */
.stProgress > div > div {
    background: var(--f1-red) !important;
}

/* ── DIVIDERS ───────────────────────────────────── */
hr {
    border-color: var(--carbon-line) !important;
    margin: 1rem 0 !important;
}

/* ── SCROLLBAR ──────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--carbon-bg); }
::-webkit-scrollbar-thumb { background: var(--f1-red); border-radius: 2px; }

/* ── CUSTOM COMPONENT CLASSES ───────────────────── */
.pw-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 0.5rem 0 1.5rem 0;
    border-bottom: 1px solid var(--carbon-line);
    margin-bottom: 1.5rem;
}
.pw-title {
    font-family: var(--font-display) !important;
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: var(--f1-white) !important;
    line-height: 1 !important;
    margin: 0 !important;
}
.pw-title span { color: var(--f1-red); }
.pw-subtitle {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    color: rgba(245,245,245,0.4) !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    margin-top: 4px !important;
}
.pw-section-label {
    font-family: var(--font-mono) !important;
    font-size: 0.65rem !important;
    color: var(--f1-red) !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    margin-bottom: 0.5rem !important;
}
.pw-card {
    background: var(--carbon-card);
    border: 1px solid var(--carbon-border);
    border-top: 2px solid var(--carbon-line);
    padding: 1.2rem 1.4rem;
    border-radius: 2px;
    margin-bottom: 0.8rem;
}
.pw-card-accent {
    border-top: 2px solid var(--f1-red);
}
.pw-card-yellow {
    border-top: 2px solid var(--f1-yellow);
}
.driver-row {
    display: flex;
    align-items: center;
    padding: 0.55rem 1rem;
    border-bottom: 1px solid var(--carbon-line);
    font-family: var(--font-ui);
    font-size: 0.95rem;
    transition: background 0.15s;
}
.driver-row:hover { background: rgba(232,0,45,0.06); }
.driver-pos {
    font-family: var(--font-display);
    font-weight: 700;
    font-size: 1.1rem;
    width: 36px;
    color: rgba(245,245,245,0.5);
}
.driver-pos-top { color: var(--f1-yellow) !important; }
.driver-name {
    font-family: var(--font-display);
    font-weight: 600;
    font-size: 1rem;
    letter-spacing: 0.05em;
    flex: 1;
}
.driver-team {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: rgba(245,245,245,0.4);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.team-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
    flex-shrink: 0;
}
.conf-bar-bg {
    height: 4px;
    background: var(--carbon-line);
    border-radius: 2px;
    width: 80px;
    margin-top: 4px;
}
.conf-bar-fill {
    height: 4px;
    border-radius: 2px;
    background: var(--f1-red);
}
.upset-badge {
    background: rgba(255,205,0,0.15);
    border: 1px solid var(--f1-yellow);
    color: var(--f1-yellow);
    font-family: var(--font-mono);
    font-size: 0.6rem;
    padding: 1px 6px;
    border-radius: 1px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.status-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: rgba(245,245,245,0.45);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.4rem 0;
}
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #2aff7a;
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}
.telemetry-line {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: rgba(245,245,245,0.35);
    letter-spacing: 0.06em;
    padding: 0.15rem 0;
    border-left: 2px solid var(--f1-red);
    padding-left: 0.7rem;
    margin-bottom: 0.25rem;
}
.info-chip {
    display: inline-block;
    background: var(--carbon-mid);
    border: 1px solid var(--carbon-border);
    font-family: var(--font-mono);
    font-size: 0.62rem;
    color: rgba(245,245,245,0.5);
    padding: 3px 8px;
    border-radius: 1px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-right: 4px;
}
.what-if-label {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--f1-yellow);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 6px;
}

/* Ensure Streamlit material icons render as icons instead of fallback text */
[data-testid="stIconMaterial"],
[class*="material-symbols"],
.material-symbols-outlined,
.material-symbols-rounded,
.material-symbols-sharp {
    font-family: 'Material Symbols Outlined' !important;
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
    font-feature-settings: 'liga' 1;
    font-size: 1.25rem !important;
    line-height: 1 !important;
    display: inline-block !important;
    text-transform: none !important;
    letter-spacing: normal !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
TEAM_COLORS = {
    "Red Bull Racing":   "#3671C6",
    "Mercedes":          "#27F4D2",
    "Ferrari":           "#E8002D",
    "McLaren":           "#FF8000",
    "Aston Martin":      "#229971",
    "Alpine":            "#FF87BC",
    "Williams":          "#64C4FF",
    "RB":                "#6692FF",
    "Haas F1 Team":      "#B6BABD",
    "Sauber":            "#52E252",
    "Kick Sauber":       "#52E252",
    "AlphaTauri":        "#5E8FAA",
    "Racing Bulls":      "#6692FF",
}

COMPOUND_COLORS = {
    "SOFT": "#e8002d",
    "MEDIUM": "#ffcd00",
    "HARD": "#f5f5f5",
    "INTERMEDIATE": "#39b54a",
    "WET": "#0067ff",
}

@st.cache_data(ttl=3600)
def load_model_artifacts():
    xgb_path = os.path.join(MODEL_DIR, "xgboost_model.pkl")
    meta_path = os.path.join(MODEL_DIR, "model_meta.pkl")
    if not os.path.exists(xgb_path):
        return None, None
    model = joblib.load(xgb_path)
    meta = joblib.load(meta_path) if os.path.exists(meta_path) else {}
    return model, meta

@st.cache_data(ttl=3600)
def load_features_df():
    path = os.path.join(PROCESSED_DIR, "features_final.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def fetch_upcoming_races():
    try:
        resp = requests.get(
            "https://api.openf1.org/v1/meetings",
            params={"year": datetime.now().year},
            timeout=10
        )
        data = resp.json()
        upcoming = [
            m for m in data
            if not m.get("is_cancelled", False)
        ]
        return sorted(upcoming, key=lambda x: x.get("date_start", ""))
    except Exception:
        return []

def fetch_current_drivers(session_key="latest"):
    try:
        resp = requests.get(
            "https://api.openf1.org/v1/drivers",
            params={"session_key": session_key},
            timeout=10
        )
        return resp.json()
    except Exception:
        return []

def get_team_color(team_name):
    for k, v in TEAM_COLORS.items():
        if k.lower() in str(team_name).lower():
            return v
    return "#555555"

def confidence_score(predicted_pos, grid_pos):
    delta = abs(predicted_pos - grid_pos)
    conf = max(0.3, 1.0 - (delta * 0.05))
    return round(conf, 2)

def predict_race(model, feature_cols, race_data_df):
    """Run prediction on a prepared dataframe of drivers for a race."""
    X = []
    for _, row in race_data_df.iterrows():
        feat = [row.get(c, 0) or 0 for c in feature_cols]
        X.append(feat)
    X = np.array(X, dtype=float)
    preds = model.predict(X)
    return preds

def build_race_input(drivers_info, grid_positions, weather_vals, champ_standings, circuit_type_enc=0):
    """Build a DataFrame for prediction from user inputs."""
    rows = []
    for d in drivers_info:
        dn = d.get("driver_number")
        grid = grid_positions.get(dn, 10)
        champ_pts = champ_standings.get(dn, {}).get("points", 100)
        champ_pos = champ_standings.get(dn, {}).get("position", 10)
        rows.append({
            "driver_number": dn,
            "name_acronym": d.get("name_acronym", "UNK"),
            "full_name": d.get("full_name", "Unknown"),
            "team_name": d.get("team_name", "Unknown"),
            "grid_position": grid,
            "quali_delta_to_pole": (grid - 1) * 0.08,
            "total_stints": 2,
            "avg_tyre_age": 2,
            "num_pit_stops": 1,
            "avg_stop_duration": 2.5,
            "air_temperature": weather_vals.get("air_temp", 25),
            "track_temperature": weather_vals.get("track_temp", 35),
            "humidity": weather_vals.get("humidity", 50),
            "wind_speed": weather_vals.get("wind_speed", 2),
            "rainfall": weather_vals.get("rainfall", 0),
            "champ_points_before": champ_pts,
            "champ_position_before": champ_pos,
            "team_dnf_rate": 0.05,
            "driver_form_3": grid * 0.9,
            "circuit_specialist_score": grid * 0.85,
            "circuit_type_enc": circuit_type_enc,
            "compound_enc": 1,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
f1_logo_path = os.path.join(BASE_DIR, "assets", "f1_logo.png")

col_logo, col_title = st.columns([1, 11])
with col_logo:
    if os.path.exists(f1_logo_path):
        st.image(f1_logo_path, width=68)
    else:
        st.markdown("<div style='font-size:2.2rem;line-height:1;padding-top:6px'>🏎</div>",
                    unsafe_allow_html=True)
with col_title:
    st.markdown("""
    <div style="padding-top:2px">
        <div class="pw-title">PITWALL <span>INTEL</span></div>
        <div class="pw-subtitle">Data Driven &nbsp;·&nbsp; Race Ready &nbsp;·&nbsp; Powered by OpenF1</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="status-bar">
    <span class="status-dot"></span>
    SYSTEM ONLINE &nbsp;·&nbsp; OPENF1 API CONNECTED &nbsp;·&nbsp;
    MODEL: XGBOOST + RANDOM FOREST
</div>
<hr/>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;
    font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
    color:#e8002d;padding-bottom:0.6rem;border-bottom:1px solid #222;
    margin-bottom:1rem">
    RACE CONTROL
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='pw-section-label'>Mission</div>", unsafe_allow_html=True)
    page = st.selectbox(
        "Select Screen",
        ["Race Predictor", "Model Performance", "Data Explorer", "What-If Simulator"],
        label_visibility="collapsed"
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("<div class='pw-section-label'>Data Pipeline</div>", unsafe_allow_html=True)

    if st.button("Fetch + Train Model", use_container_width=True):
        st.session_state["run_pipeline"] = True

    features_exist = os.path.exists(os.path.join(PROCESSED_DIR, "features_final.csv"))
    model_exists = os.path.exists(os.path.join(MODEL_DIR, "xgboost_model.pkl"))

    st.markdown(f"""
    <div style="margin-top:0.8rem">
        <div class="telemetry-line">DATA &nbsp;{'LOADED' if features_exist else 'NOT FOUND'}</div>
        <div class="telemetry-line">MODEL {'READY' if model_exists else 'NOT TRAINED'}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("<div class='pw-section-label'>About</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;
    color:rgba(245,245,245,0.3);line-height:1.6;letter-spacing:0.04em">
    SOURCE &nbsp;: OPENF1.ORG<br>
    SEASONS : 2023 / 2024 / 2025<br>
    MODEL &nbsp; : XGBOOST v2.0<br>
    BUILD &nbsp; : PITWALL INTEL v1.0
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────
if st.session_state.get("run_pipeline"):
    st.session_state["run_pipeline"] = False
    with st.spinner(""):
        progress_bar = st.progress(0)
        status_text = st.empty()

        steps = [
            (10, "Connecting to OpenF1 API..."),
            (20, "Fetching race sessions 2023-2025..."),
            (35, "Downloading race results..."),
            (45, "Pulling qualifying grids..."),
            (55, "Loading tyre strategy data..."),
            (65, "Processing pit stop data..."),
            (72, "Fetching weather conditions..."),
            (80, "Loading championship standings..."),
            (88, "Engineering features..."),
            (94, "Training XGBoost model..."),
            (100, "Pipeline complete. Model ready."),
        ]

        try:
            from utils.data_pipeline import run_pipeline
            from utils.model_trainer import train_and_save

            for pct, msg in steps[:9]:
                status_text.markdown(f"""
                <div class="telemetry-line">{msg}</div>
                """, unsafe_allow_html=True)
                progress_bar.progress(pct)
                time.sleep(0.2)

            df = run_pipeline(years=[2023, 2024, 2025], save=True)

            progress_bar.progress(94)
            status_text.markdown("""
            <div class="telemetry-line">Training XGBoost model...</div>
            """, unsafe_allow_html=True)

            train_and_save()

            progress_bar.progress(100)
            status_text.markdown("""
            <div class="telemetry-line" style="color:#2aff7a">Pipeline complete. Model ready.</div>
            """, unsafe_allow_html=True)
            st.success("Model trained and ready.")
            st.cache_data.clear()

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.info("Check your internet connection and try again.")


# ─────────────────────────────────────────────────────────────
# MAIN PAGES
# ─────────────────────────────────────────────────────────────

# ── PAGE 1: RACE PREDICTOR ───────────────────────────────────
if page == "Race Predictor":
    model, meta = load_model_artifacts()

    if not model_exists or model is None:
        st.markdown("""
        <div class="pw-card pw-card-accent">
            <div class="pw-section-label">No Model Detected</div>
            <div style="font-family:'Rajdhani',sans-serif;font-size:1rem;
            color:rgba(245,245,245,0.6);line-height:1.7">
                Click <strong style="color:#e8002d">Fetch + Train Model</strong>
                in the sidebar to download F1 data and train the prediction model.
                This will pull 3 seasons of race data from OpenF1 and train
                an XGBoost model. Estimated time: 3 to 8 minutes.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="pw-card">
                <div class="pw-section-label">Step 01</div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;
                font-weight:600;letter-spacing:0.05em">Fetch Race Data</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;
                color:rgba(245,245,245,0.4);margin-top:6px;line-height:1.6">
                Sessions · Results · Grids<br>Stints · Pit Stops · Weather
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="pw-card">
                <div class="pw-section-label">Step 02</div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;
                font-weight:600;letter-spacing:0.05em">Engineer Features</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;
                color:rgba(245,245,245,0.4);margin-top:6px;line-height:1.6">
                Driver Form · Quali Delta<br>Tyre Strategy · Constructor Rate
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="pw-card pw-card-accent">
                <div class="pw-section-label">Step 03</div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;
                font-weight:600;letter-spacing:0.05em">Train XGBoost</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;
                color:rgba(245,245,245,0.4);margin-top:6px;line-height:1.6">
                XGBoost + Random Forest<br>SHAP Explainability
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("<div class='pw-section-label'>Race Setup</div>", unsafe_allow_html=True)

        col_race, col_weather = st.columns([2, 1])

        with col_race:
            with st.spinner("Loading upcoming races..."):
                upcoming = fetch_upcoming_races()

            race_names = [
                f"{m.get('meeting_name','?')} — {m.get('date_start','')[:10]}"
                for m in upcoming
            ] if upcoming else ["Bahrain Grand Prix — 2025-03-02"]

            selected_race_label = st.selectbox("Select Grand Prix", race_names)
            selected_meeting = upcoming[race_names.index(selected_race_label)] if upcoming else {}

            circuit_type_raw = selected_meeting.get("circuit_type", "Permanent")
            circuit_type_enc = 1 if "Street" in circuit_type_raw else (2 if "Road" in circuit_type_raw else 0)

        with col_weather:
            st.markdown("<div class='pw-section-label'>Weather Conditions</div>", unsafe_allow_html=True)
            rainfall = st.selectbox("Conditions", ["Dry", "Wet", "Mixed"])
            air_temp = st.slider("Air Temp (C)", 10, 45, 24)
            track_temp = st.slider("Track Temp (C)", 15, 65, 38)

        weather_vals = {
            "air_temp": air_temp,
            "track_temp": track_temp,
            "humidity": 55,
            "wind_speed": 2.5,
            "rainfall": 1 if rainfall == "Wet" else 0,
        }

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("<div class='pw-section-label'>Grid Setup</div>", unsafe_allow_html=True)

        with st.spinner("Loading current drivers..."):
            raw_drivers = fetch_current_drivers("latest")
            if not raw_drivers:
                raw_drivers = [
                    {"driver_number": 1,  "name_acronym": "VER", "full_name": "Max Verstappen",     "team_name": "Red Bull Racing"},
                    {"driver_number": 63, "name_acronym": "RUS", "full_name": "George Russell",     "team_name": "Mercedes"},
                    {"driver_number": 44, "name_acronym": "HAM", "full_name": "Lewis Hamilton",     "team_name": "Ferrari"},
                    {"driver_number": 16, "name_acronym": "LEC", "full_name": "Charles Leclerc",    "team_name": "Ferrari"},
                    {"driver_number": 4,  "name_acronym": "NOR", "full_name": "Lando Norris",       "team_name": "McLaren"},
                    {"driver_number": 81, "name_acronym": "PIA", "full_name": "Oscar Piastri",      "team_name": "McLaren"},
                    {"driver_number": 14, "name_acronym": "ALO", "full_name": "Fernando Alonso",    "team_name": "Aston Martin"},
                    {"driver_number": 18, "name_acronym": "STR", "full_name": "Lance Stroll",       "team_name": "Aston Martin"},
                    {"driver_number": 10, "name_acronym": "GAS", "full_name": "Pierre Gasly",       "team_name": "Alpine"},
                    {"driver_number": 31, "name_acronym": "OCO", "full_name": "Esteban Ocon",       "team_name": "Alpine"},
                    {"driver_number": 23, "name_acronym": "ALB", "full_name": "Alexander Albon",    "team_name": "Williams"},
                    {"driver_number": 2,  "name_acronym": "SAR", "full_name": "Logan Sargeant",     "team_name": "Williams"},
                    {"driver_number": 22, "name_acronym": "TSU", "full_name": "Yuki Tsunoda",       "team_name": "Racing Bulls"},
                    {"driver_number": 3,  "name_acronym": "RIC", "full_name": "Daniel Ricciardo",   "team_name": "Racing Bulls"},
                    {"driver_number": 20, "name_acronym": "MAG", "full_name": "Kevin Magnussen",    "team_name": "Haas F1 Team"},
                    {"driver_number": 27, "name_acronym": "HUL", "full_name": "Nico Hulkenberg",    "team_name": "Haas F1 Team"},
                    {"driver_number": 77, "name_acronym": "BOT", "full_name": "Valtteri Bottas",    "team_name": "Sauber"},
                    {"driver_number": 24, "name_acronym": "ZHO", "full_name": "Guanyu Zhou",        "team_name": "Sauber"},
                    {"driver_number": 55, "name_acronym": "SAI", "full_name": "Carlos Sainz",       "team_name": "Ferrari"},
                    {"driver_number": 11, "name_acronym": "PER", "full_name": "Sergio Perez",       "team_name": "Red Bull Racing"},
                ]

        seen_numbers = set()
        unique_drivers = []
        for d in raw_drivers:
            if d.get("driver_number") not in seen_numbers:
                seen_numbers.add(d.get("driver_number"))
                unique_drivers.append(d)

        unique_drivers = unique_drivers[:20]

        driver_names = [
            f"P{i+1:02d} — {d.get('name_acronym','?')} ({d.get('team_name','?')})"
            for i, d in enumerate(unique_drivers)
        ]

        with st.expander("Customize Starting Grid (optional)", expanded=False):
            st.markdown("""
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;
            color:rgba(245,245,245,0.4);margin-bottom:0.8rem">
            Drag the sliders to set each driver's grid position for the prediction.
            </div>
            """, unsafe_allow_html=True)
            grid_positions = {}
            cols = st.columns(4)
            for i, d in enumerate(unique_drivers[:20]):
                with cols[i % 4]:
                    pos = st.number_input(
                        d.get("name_acronym", "?"),
                        min_value=1, max_value=20,
                        value=i + 1, key=f"grid_{d['driver_number']}"
                    )
                    grid_positions[d["driver_number"]] = pos

        if not grid_positions:
            grid_positions = {d["driver_number"]: i + 1 for i, d in enumerate(unique_drivers)}

        champ_standings = {
            d["driver_number"]: {
                "points": max(0, 400 - grid_positions.get(d["driver_number"], i+1) * 18),
                "position": grid_positions.get(d["driver_number"], i+1),
            }
            for i, d in enumerate(unique_drivers)
        }

        st.markdown("<hr/>", unsafe_allow_html=True)

        if st.button("RUN PREDICTION", use_container_width=True):
            with st.spinner("Analysing race data..."):
                time.sleep(0.5)

            race_df = build_race_input(
                unique_drivers, grid_positions, weather_vals,
                champ_standings, circuit_type_enc
            )

            feature_cols = meta.get("feature_cols", [
                "grid_position", "quali_delta_to_pole", "total_stints",
                "avg_tyre_age", "num_pit_stops", "avg_stop_duration",
                "air_temperature", "track_temperature", "humidity",
                "wind_speed", "rainfall", "champ_points_before",
                "champ_position_before", "team_dnf_rate", "driver_form_3",
                "circuit_specialist_score", "circuit_type_enc", "compound_enc",
            ])

            preds = predict_race(model, feature_cols, race_df)
            race_df["predicted_score"] = preds
            race_df = race_df.sort_values("predicted_score").reset_index(drop=True)
            race_df["predicted_position"] = range(1, len(race_df) + 1)

            grid_map = {d["driver_number"]: grid_positions.get(d["driver_number"], i+1)
                        for i, d in enumerate(unique_drivers)}
            race_df["grid_pos"] = race_df["driver_number"].map(grid_map)
            race_df["position_gain"] = race_df["grid_pos"] - race_df["predicted_position"]

            st.session_state["prediction_result"] = race_df
            st.session_state["selected_race_label"] = selected_race_label

        if "prediction_result" in st.session_state:
            result_df = st.session_state["prediction_result"]
            race_label = st.session_state.get("selected_race_label", "")

            st.markdown(f"""
            <div style="margin:1.5rem 0 0.8rem 0">
                <div class="pw-section-label">Predicted Race Classification</div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.5rem;
                font-weight:700;letter-spacing:0.06em;color:var(--f1-white)">
                {race_label.split('—')[0].strip().upper()}
                </div>
            </div>
            """, unsafe_allow_html=True)

            tab_grid, tab_gain, tab_podium, tab_upset = st.tabs([
                "Predicted Grid", "Position Change", "Podium Breakdown", "Upset Alert"
            ])

            with tab_grid:
                grid_html = """
                <style>
                body {
                    margin: 0;
                    background: transparent;
                    color: #f5f5f5;
                    font-family: 'Rajdhani', sans-serif;
                }
                .grid-wrap {
                    border: 1px solid #222;
                    border-radius: 2px;
                    overflow: hidden;
                    border-top: 2px solid #e8002d;
                    background: #0b0b0b;
                    color: #f5f5f5;
                }
                .grid-head {
                    display: flex;
                    align-items: center;
                    padding: 0.6rem 1rem;
                    background: #111;
                    font-family: 'Share Tech Mono', monospace;
                    font-size: 0.62rem;
                    color: rgba(245,245,245,0.75);
                    letter-spacing: 0.1em;
                    text-transform: uppercase;
                    border-bottom: 1px solid #222;
                }
                .driver-row {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    padding: 0.7rem 1rem;
                    margin: 0;
                    border-bottom: 1px solid rgba(255,255,255,0.05);
                    background: rgba(255,255,255,0.03);
                    backdrop-filter: blur(6px);
                    -webkit-backdrop-filter: blur(6px);
                    transition: background 0.15s ease;
                }
                .driver-row:nth-child(even) {
                    background: rgba(255,255,255,0.05);
                }
                .driver-row:hover {
                    background: rgba(232,0,45,0.10);
                }
                .driver-pos {
                    width: 24px;
                    flex: 0 0 24px;
                    text-align: right;
                    font-family: 'Barlow Condensed', sans-serif;
                    font-size: 0.92rem;
                    font-weight: 700;
                    color: rgba(245,245,245,0.60);
                }
                .driver-pos-top {
                    color: #ffcd00;
                }
                .team-dot {
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    flex: 0 0 8px;
                }
                .driver-name {
                    flex: 1;
                    font-family: 'Barlow Condensed', sans-serif;
                    font-size: 1.02rem;
                    font-weight: 800;
                    letter-spacing: 0.05em;
                    color: #f5f5f5;
                }
                .driver-team {
                    width: 140px;
                    flex: 0 0 140px;
                    font-family: 'Share Tech Mono', monospace;
                    font-size: 0.64rem;
                    color: rgba(245,245,245,0.78);
                    letter-spacing: 0.06em;
                    text-transform: uppercase;
                }
                .driver-grid {
                    width: 48px;
                    flex: 0 0 48px;
                    text-align: right;
                    font-family: 'Share Tech Mono', monospace;
                    font-size: 0.70rem;
                    color: rgba(245,245,245,0.72);
                }
                .driver-delta {
                    width: 56px;
                    flex: 0 0 56px;
                    text-align: right;
                    font-family: 'Share Tech Mono', monospace;
                    font-size: 0.70rem;
                    font-weight: 700;
                }
                .upset-badge {
                    background: rgba(255,205,0,0.15);
                    border: 1px solid #ffcd00;
                    color: #ffcd00;
                    font-family: 'Share Tech Mono', monospace;
                    font-size: 0.58rem;
                    padding: 1px 6px;
                    border-radius: 1px;
                    letter-spacing: 0.1em;
                    text-transform: uppercase;
                    margin-left: 8px;
                }
                </style>
                <div class="grid-wrap">
                """
                grid_html += """
                <div class="grid-head">
                    <span style="width:24px;text-align:right">POS</span>
                    <span style="width:12px;margin-right:0"></span>
                    <span style="flex:1">DRIVER</span>
                    <span style="width:140px">TEAM</span>
                    <span style="width:48px;text-align:right">GRID</span>
                    <span style="width:56px;text-align:right">DELTA</span>
                </div>
                """

                for _, row in result_df.iterrows():
                    pos = int(row["predicted_position"])
                    pos_cls = "driver-pos-top" if pos <= 3 else ""
                    team_col = get_team_color(row.get("team_name", ""))
                    grid_p = int(row.get("grid_pos", pos))
                    delta = int(row.get("position_gain", 0))
                    delta_str = f"+{delta}" if delta > 0 else str(delta)
                    delta_col = "#2aff7a" if delta > 0 else ("#e8002d" if delta < 0 else "rgba(245,245,245,0.4)")

                    conf = confidence_score(pos, grid_p)
                    conf_w = int(conf * 80)

                    is_upset = pos <= 5 and grid_p >= 8
                    upset_html = '<span class="upset-badge" style="margin-left:8px">UPSET</span>' if is_upset else ""

                    grid_html += f"""
                    <div class="driver-row">
                        <span class="driver-pos {pos_cls}">{pos:02d}</span>
                        <span class="team-dot" style="background:{team_col}"></span>
                        <span class="driver-name">{row.get('name_acronym','?')}{upset_html}</span>
                        <span class="driver-team">{row.get('team_name','?')[:18]}</span>
                        <span class="driver-grid">P{grid_p}</span>
                        <span class="driver-delta" style="color:{delta_col}">{delta_str}</span>
                    </div>
                    """

                grid_html += "</div>"
                grid_html += "</div>"
                grid_height = max(260, 62 + (len(result_df) * 44))
                components.html(grid_html, height=grid_height, scrolling=True)

            with tab_gain:
                fig = go.Figure()
                colors = [get_team_color(r["team_name"]) for _, r in result_df.iterrows()]
                fig.add_trace(go.Bar(
                    x=result_df["name_acronym"],
                    y=result_df["position_gain"],
                    marker_color=colors,
                    marker_line_width=0,
                    text=[f"+{v}" if v > 0 else str(v) for v in result_df["position_gain"]],
                    textposition="outside",
                    textfont=dict(family="Share Tech Mono", size=10, color="#f5f5f5"),
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Rajdhani", color="#f5f5f5"),
                    xaxis=dict(
                        gridcolor="#1a1a1a", tickfont=dict(family="Barlow Condensed", size=11),
                        title=dict(text="DRIVER", font=dict(family="Share Tech Mono", size=10, color="rgba(245,245,245,0.4)")),
                    ),
                    yaxis=dict(
                        gridcolor="#1a1a1a",
                        title=dict(text="POSITION GAIN / LOSS", font=dict(family="Share Tech Mono", size=10, color="rgba(245,245,245,0.4)")),
                        zerolinecolor="#333",
                    ),
                    margin=dict(l=40, r=20, t=20, b=40),
                    height=360,
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab_podium:
                top3 = result_df.head(3)
                c1, c2, c3 = st.columns(3)
                cols_podium = [c1, c2, c3]
                medals = ["01", "02", "03"]
                for i, (_, row) in enumerate(top3.iterrows()):
                    with cols_podium[i]:
                        team_col = get_team_color(row.get("team_name", ""))
                        st.markdown(f"""
                        <div class="pw-card" style="border-top:3px solid {team_col};text-align:center;padding:1.4rem 1rem">
                            <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;
                            color:rgba(245,245,245,0.35);letter-spacing:0.12em;margin-bottom:6px">
                            P{medals[i]}</div>
                            <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.8rem;
                            font-weight:800;letter-spacing:0.06em;color:#f5f5f5">
                            {row.get('name_acronym','?')}</div>
                            <div style="font-family:'Rajdhani',sans-serif;font-size:0.85rem;
                            color:rgba(245,245,245,0.5);margin-top:2px">
                            {row.get('team_name','?')}</div>
                            <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
                            color:rgba(245,245,245,0.3);margin-top:8px">
                            Started P{int(row.get('grid_pos', i+1))}</div>
                        </div>
                        """, unsafe_allow_html=True)

            with tab_upset:
                upsets = result_df[(result_df["predicted_position"] <= 5) & (result_df["grid_pos"] >= 8)]
                if upsets.empty:
                    st.markdown("""
                    <div class="pw-card">
                        <div style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;
                        color:rgba(245,245,245,0.4);text-align:center;padding:1rem">
                        No significant upsets predicted for this race.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    for _, row in upsets.iterrows():
                        team_col = get_team_color(row.get("team_name", ""))
                        st.markdown(f"""
                        <div class="pw-card pw-card-yellow">
                            <div style="display:flex;align-items:center;gap:12px">
                                <span class="team-dot" style="background:{team_col};width:12px;height:12px"></span>
                                <div>
                                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.2rem;
                                    font-weight:700;letter-spacing:0.06em">
                                    {row.get('name_acronym','?')} — PREDICTED P{int(row['predicted_position'])}
                                    </div>
                                    <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;
                                    color:rgba(245,245,245,0.45);margin-top:3px">
                                    Starting from P{int(row['grid_pos'])} — {int(row['position_gain'])} position gain predicted
                                    </div>
                                </div>
                                <span class="upset-badge" style="margin-left:auto">UPSET ALERT</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)


# ── PAGE 2: MODEL PERFORMANCE ────────────────────────────────
elif page == "Model Performance":
    model, meta = load_model_artifacts()
    df = load_features_df()

    st.markdown("<div class='pw-section-label'>Model Metrics</div>", unsafe_allow_html=True)

    if meta:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Best Model", meta.get("best_model", "XGBoost"))
        with c2:
            st.metric("XGBoost MAE", f"{meta.get('xgb_mae', 0):.2f} pos")
        with c3:
            st.metric("RF MAE", f"{meta.get('rf_mae', 0):.2f} pos")
        with c4:
            podium = meta.get("xgb_podium_acc", 0)
            st.metric("Podium Accuracy", f"{podium*100:.1f}%")
    else:
        st.info("Train the model first to see performance metrics.")

    if model and hasattr(model, "feature_importances_"):
        feature_cols = meta.get("feature_cols", []) if meta else []
        if feature_cols:
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown("<div class='pw-section-label'>Feature Importance</div>", unsafe_allow_html=True)
            imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
            fig = go.Figure(go.Bar(
                x=imp.values,
                y=imp.index,
                orientation="h",
                marker_color=[
                    "#e8002d" if v == imp.max() else
                    "#ffcd00" if v >= imp.quantile(0.8) else "#333"
                    for v in imp.values
                ],
                marker_line_width=0,
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Rajdhani", color="#f5f5f5"),
                xaxis=dict(gridcolor="#1a1a1a", title=dict(
                    text="IMPORTANCE SCORE",
                    font=dict(family="Share Tech Mono", size=9, color="rgba(245,245,245,0.4)")
                )),
                yaxis=dict(tickfont=dict(family="Share Tech Mono", size=10)),
                margin=dict(l=10, r=20, t=10, b=40),
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)


# ── PAGE 3: DATA EXPLORER ────────────────────────────────────
elif page == "Data Explorer":
    df = load_features_df()

    st.markdown("<div class='pw-section-label'>Dataset Overview</div>", unsafe_allow_html=True)

    if df is None:
        st.markdown("""
        <div class="pw-card pw-card-accent">
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.8rem;
            color:rgba(245,245,245,0.5)">
            No data found. Run the pipeline from the sidebar first.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Rows", f"{len(df):,}")
        with c2:
            st.metric("Seasons", df["year"].nunique() if "year" in df else "N/A")
        with c3:
            st.metric("Race Events", df["circuit_short_name"].nunique() if "circuit_short_name" in df else "N/A")
        with c4:
            st.metric("Features", len(df.columns))

        st.markdown("<hr/>", unsafe_allow_html=True)

        if "year" in df.columns and "circuit_short_name" in df.columns:
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                yr = st.selectbox("Season", sorted(df["year"].unique(), reverse=True))
            with col_f2:
                circuits = ["All"] + sorted(df[df["year"] == yr]["circuit_short_name"].unique().tolist())
                circ = st.selectbox("Circuit", circuits)

            filtered = df[df["year"] == yr]
            if circ != "All":
                filtered = filtered[filtered["circuit_short_name"] == circ]

            display_cols = [c for c in [
                "name_acronym", "team_name", "circuit_short_name",
                "grid_position", "finish_position", "quali_delta_to_pole",
                "driver_form_3", "rainfall", "total_stints"
            ] if c in filtered.columns]

            st.dataframe(
                filtered[display_cols].reset_index(drop=True),
                use_container_width=True,
                height=380,
            )

        # Position gain distribution
        if "grid_position" in df.columns and "finish_position" in df.columns:
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown("<div class='pw-section-label'>Grid vs Finish Position</div>", unsafe_allow_html=True)
            sample = df.dropna(subset=["grid_position", "finish_position"])
            if "team_name" in sample.columns:
                sample["team_color_hex"] = sample["team_name"].apply(get_team_color)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=sample["grid_position"],
                y=sample["finish_position"],
                mode="markers",
                marker=dict(
                    color=sample["team_color_hex"] if "team_color_hex" in sample.columns else "#e8002d",
                    size=5, opacity=0.7, line=dict(width=0)
                ),
                text=sample.get("name_acronym", ""),
                hovertemplate="<b>%{text}</b><br>Grid: %{x}<br>Finish: %{y}<extra></extra>",
            ))
            fig2.add_trace(go.Scatter(
                x=[1, 20], y=[1, 20], mode="lines",
                line=dict(color="#333", dash="dot", width=1),
                showlegend=False
            ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Rajdhani", color="#f5f5f5"),
                xaxis=dict(
                    gridcolor="#1a1a1a",
                    title=dict(text="GRID POSITION", font=dict(family="Share Tech Mono", size=9, color="rgba(245,245,245,0.4)"))
                ),
                yaxis=dict(
                    gridcolor="#1a1a1a",
                    title=dict(text="FINISH POSITION", font=dict(family="Share Tech Mono", size=9, color="rgba(245,245,245,0.4)"))
                ),
                margin=dict(l=40, r=20, t=10, b=40),
                height=380,
            )
            st.plotly_chart(fig2, use_container_width=True)


# ── PAGE 4: WHAT-IF SIMULATOR ────────────────────────────────
elif page == "What-If Simulator":
    model, meta = load_model_artifacts()

    st.markdown("""
    <div class="pw-card pw-card-yellow" style="margin-bottom:1.5rem">
        <div class="pw-section-label">What-If Simulator</div>
        <div style="font-family:'Rajdhani',sans-serif;font-size:0.95rem;
        color:rgba(245,245,245,0.6);line-height:1.6">
        Change grid position, weather, or circuit conditions in real time
        and see how the prediction shifts. Runs the model live on every change.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not model_exists or model is None:
        st.info("Train the model first to use the What-If Simulator.")
    else:
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("<div class='what-if-label'>Driver Settings</div>", unsafe_allow_html=True)
            driver_name = st.selectbox("Focus Driver", [
                "VER", "HAM", "LEC", "NOR", "PIA", "RUS",
                "ALO", "SAI", "GAS", "TSU", "ALB", "HUL"
            ])
            grid_pos = st.slider("Grid Position", 1, 20, 5)
            champ_pts = st.slider("Championship Points", 0, 450, 200)
            champ_pos_val = st.slider("Championship Position", 1, 20, 4)

        with col_right:
            st.markdown("<div class='what-if-label'>Race Conditions</div>", unsafe_allow_html=True)
            rain_wi = st.selectbox("Rainfall", ["Dry", "Wet"])
            air_t = st.slider("Air Temperature (C)", 10, 45, 24)
            track_t = st.slider("Track Temperature (C)", 15, 65, 38)
            circuit_wi = st.selectbox("Circuit Type", ["Permanent", "Street", "Road"])

        circuit_enc = {"Permanent": 0, "Street": 1, "Road": 2}[circuit_wi]
        feature_cols = meta.get("feature_cols") if meta else [
            "grid_position", "quali_delta_to_pole", "total_stints",
            "avg_tyre_age", "num_pit_stops", "avg_stop_duration",
            "air_temperature", "track_temperature", "humidity",
            "wind_speed", "rainfall", "champ_points_before",
            "champ_position_before", "team_dnf_rate", "driver_form_3",
            "circuit_specialist_score", "circuit_type_enc", "compound_enc",
        ]

        input_vals = {
            "grid_position": grid_pos,
            "quali_delta_to_pole": (grid_pos - 1) * 0.08,
            "total_stints": 2,
            "avg_tyre_age": 2,
            "num_pit_stops": 1,
            "avg_stop_duration": 2.5,
            "air_temperature": air_t,
            "track_temperature": track_t,
            "humidity": 55,
            "wind_speed": 2,
            "rainfall": 1 if rain_wi == "Wet" else 0,
            "champ_points_before": champ_pts,
            "champ_position_before": champ_pos_val,
            "team_dnf_rate": 0.05,
            "driver_form_3": grid_pos * 0.9,
            "circuit_specialist_score": grid_pos * 0.85,
            "circuit_type_enc": circuit_enc,
            "compound_enc": 1,
        }

        X = np.array([[input_vals.get(c, 0) for c in feature_cols]])
        pred = model.predict(X)[0]
        pred_pos = max(1, min(20, round(pred)))

        st.markdown("<hr/>", unsafe_allow_html=True)

        pos_color = "#ffcd00" if pred_pos <= 3 else ("#2aff7a" if pred_pos <= 6 else "#e8002d" if pred_pos >= 15 else "#f5f5f5")
        delta_from_grid = grid_pos - pred_pos
        delta_str = f"+{delta_from_grid}" if delta_from_grid > 0 else str(delta_from_grid)

        st.markdown(f"""
        <div class="pw-card pw-card-accent" style="text-align:center;padding:2rem">
            <div class="pw-section-label" style="text-align:center">{driver_name} — Predicted Outcome</div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:5rem;
            font-weight:800;letter-spacing:0.04em;color:{pos_color};line-height:1">
            P{pred_pos:02d}
            </div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.8rem;
            color:rgba(245,245,245,0.4);margin-top:8px">
            STARTED P{grid_pos:02d} &nbsp;·&nbsp;
            <span style="color:{pos_color}">{delta_str} POSITIONS</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Sensitivity chart: grid pos 1-20 vs predicted finish
        st.markdown("<div class='pw-section-label' style='margin-top:1.5rem'>Position Sensitivity</div>",
                    unsafe_allow_html=True)
        grid_range = list(range(1, 21))
        preds_range = []
        for gp in grid_range:
            vals = dict(input_vals)
            vals["grid_position"] = gp
            vals["quali_delta_to_pole"] = (gp - 1) * 0.08
            vals["driver_form_3"] = gp * 0.9
            vals["circuit_specialist_score"] = gp * 0.85
            Xr = np.array([[vals.get(c, 0) for c in feature_cols]])
            pr = max(1, min(20, round(model.predict(Xr)[0])))
            preds_range.append(pr)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=grid_range, y=preds_range, mode="lines+markers",
            line=dict(color="#e8002d", width=2),
            marker=dict(color="#ffcd00", size=7, line=dict(color="#e8002d", width=1)),
            fill="tozeroy", fillcolor="rgba(232,0,45,0.07)",
        ))
        fig3.add_trace(go.Scatter(
            x=[grid_pos], y=[pred_pos], mode="markers",
            marker=dict(color="#ffcd00", size=12, symbol="diamond",
                        line=dict(color="#fff", width=1)),
            showlegend=False, name="Current",
        ))
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Rajdhani", color="#f5f5f5"),
            xaxis=dict(
                gridcolor="#1a1a1a", dtick=2,
                title=dict(text="GRID POSITION", font=dict(family="Share Tech Mono", size=9, color="rgba(245,245,245,0.4)"))
            ),
            yaxis=dict(
                gridcolor="#1a1a1a", autorange="reversed",
                title=dict(text="PREDICTED FINISH", font=dict(family="Share Tech Mono", size=9, color="rgba(245,245,245,0.4)"))
            ),
            showlegend=False,
            margin=dict(l=40, r=20, t=10, b=40),
            height=300,
        )
        st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("""
<div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;
color:rgba(245,245,245,0.2);letter-spacing:0.1em;text-align:center;padding:0.5rem 0">
PITWALL INTEL v1.0 &nbsp;·&nbsp; DATA: OPENF1.ORG &nbsp;·&nbsp;
MODEL: XGBOOST + RANDOM FOREST &nbsp;·&nbsp; FOR EDUCATIONAL USE ONLY
</div>
""", unsafe_allow_html=True)
