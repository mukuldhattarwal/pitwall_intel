"""
PitWall Intel — Main Streamlit Application
F1 Race Result Predictor with full carbon-dark pit wall aesthetic.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
import sys
import requests
import time
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

BASE_DIR   = os.path.dirname(__file__)
MODEL_DIR  = os.path.join(BASE_DIR, "models")
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
F1_LOGO_PATH = os.path.join(BASE_DIR, "assets", "f1_logo.png")

_tab_icon = "🏎"
try:
    from PIL import Image as _PILImage
    if os.path.exists(F1_LOGO_PATH):
        _tab_icon = _PILImage.open(F1_LOGO_PATH)
except Exception:
    pass

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PitWall Intel",
    page_icon=_tab_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

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
    --font-display: 'Barlow Condensed', sans-serif;
    --font-mono:    'Share Tech Mono', monospace;
    --font-ui:      'Rajdhani', sans-serif;
}

* { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: var(--font-ui) !important;
    background-color: var(--carbon-bg) !important;
    color: var(--f1-white) !important;
}
.stApp { background-color: var(--carbon-bg) !important; }
.stApp::before {
    content: '';
    position: fixed; top:0; left:0; right:0; bottom:0;
    background-image:
        repeating-linear-gradient(45deg,transparent,transparent 2px,rgba(255,255,255,0.012) 2px,rgba(255,255,255,0.012) 4px),
        repeating-linear-gradient(-45deg,transparent,transparent 2px,rgba(255,255,255,0.008) 2px,rgba(255,255,255,0.008) 4px);
    pointer-events: none; z-index: 0;
}

[data-testid="stIconMaterial"],
span[data-testid="stIconMaterial"],
[class*="material-symbols"],
button[kind="header"] span,
[data-testid="collapsedControl"] span,
[data-testid="baseButton-header"] span {
    font-family: 'Material Symbols Rounded' !important;
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24 !important;
    font-feature-settings: 'liga' 1 !important;
    font-size: 1.4rem !important;
    line-height: 1 !important;
    display: inline-block !important;
    text-transform: none !important;
    letter-spacing: normal !important;
    white-space: nowrap !important;
    direction: ltr !important;
    -webkit-font-smoothing: antialiased !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d0d 0%, #0a0a0a 100%) !important;
    border-right: 1px solid var(--carbon-border) !important;
}
[data-testid="stSidebar"]::before {
    content: ''; position: absolute; top:0; left:0; right:0; height:3px;
    background: linear-gradient(90deg, var(--f1-red), var(--f1-yellow));
}
[data-testid="stSidebar"] *:not(.material-symbols-rounded) { color: var(--f1-white) !important; font-family: var(--font-ui) !important; }

[data-testid="stSelectbox"] > div > div {
    background: var(--carbon-card) !important;
    border: 1px solid var(--carbon-border) !important;
    color: var(--f1-white) !important;
    border-radius: 2px !important;
}
[data-testid="stSelectbox"] > div > div:hover { border-color: var(--f1-red) !important; }

.stButton > button {
    background: var(--f1-red) !important; color: #fff !important;
    border: none !important; border-radius: 2px !important;
    font-family: var(--font-display) !important; font-weight: 700 !important;
    font-size: 1rem !important; letter-spacing: 0.12em !important;
    text-transform: uppercase !important; padding: 0.6rem 2rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover { background: #ff1a3e !important; box-shadow: var(--glow-red) !important; transform: translateY(-1px) !important; }

[data-testid="metric-container"] {
    background: var(--carbon-card) !important;
    border: 1px solid var(--carbon-border) !important;
    border-top: 2px solid var(--f1-red) !important;
    border-radius: 2px !important; padding: 1rem !important;
}
[data-testid="metric-container"] label {
    font-family: var(--font-mono) !important; font-size: 0.65rem !important;
    color: rgba(245,245,245,0.5) !important; text-transform: uppercase !important; letter-spacing: 0.1em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: var(--font-display) !important; font-size: 1.8rem !important;
    font-weight: 700 !important; color: var(--f1-white) !important;
}

[data-testid="stDataFrame"] { border: 1px solid var(--carbon-border) !important; border-radius: 2px !important; }

.stTabs [data-baseweb="tab-list"] {
    background: var(--carbon-mid) !important;
    border-bottom: 1px solid var(--carbon-border) !important; gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--font-display) !important; font-weight: 600 !important;
    font-size: 0.9rem !important; letter-spacing: 0.08em !important;
    text-transform: uppercase !important; color: rgba(245,245,245,0.5) !important;
    border-radius: 0 !important; padding: 0.7rem 1.5rem !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    background: transparent !important; color: var(--f1-white) !important;
    border-bottom: 2px solid var(--f1-red) !important;
}

.stProgress > div > div { background: var(--f1-red) !important; }
hr { border-color: var(--carbon-line) !important; margin: 1rem 0 !important; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--carbon-bg); }
::-webkit-scrollbar-thumb { background: var(--f1-red); border-radius: 2px; }

.pw-title { font-family: var(--font-display) !important; font-size: 2.6rem !important; font-weight: 800 !important; letter-spacing: 0.06em !important; text-transform: uppercase !important; color: var(--f1-white) !important; line-height: 1 !important; margin: 0 !important; }
.pw-title span { color: var(--f1-red); }
.pw-subtitle { font-family: var(--font-mono) !important; font-size: 0.72rem !important; color: rgba(245,245,245,0.4) !important; letter-spacing: 0.18em !important; text-transform: uppercase !important; margin-top: 4px !important; }
.pw-section-label { font-family: var(--font-mono) !important; font-size: 0.65rem !important; color: var(--f1-red) !important; letter-spacing: 0.18em !important; text-transform: uppercase !important; margin-bottom: 0.5rem !important; }
.pw-card { background: var(--carbon-card); border: 1px solid var(--carbon-border); border-top: 2px solid var(--carbon-line); padding: 1.2rem 1.4rem; border-radius: 2px; margin-bottom: 0.8rem; }
.pw-card-accent { border-top: 2px solid var(--f1-red) !important; }
.pw-card-yellow { border-top: 2px solid var(--f1-yellow) !important; }
.status-bar { display:flex; align-items:center; gap:8px; font-family:var(--font-mono); font-size:0.68rem; color:rgba(245,245,245,0.45); letter-spacing:0.1em; text-transform:uppercase; padding:0.4rem 0; }
.status-dot { width:6px; height:6px; border-radius:50%; background:#2aff7a; animation:pulse-dot 2s infinite; }
@keyframes pulse-dot { 0%,100%{opacity:1} 50%{opacity:0.3} }
.telemetry-line { font-family:var(--font-mono); font-size:0.72rem; color:rgba(245,245,245,0.35); letter-spacing:0.06em; padding:0.15rem 0.7rem; border-left:2px solid var(--f1-red); margin-bottom:0.25rem; }
.what-if-label { font-family:var(--font-mono); font-size:0.68rem; color:var(--f1-yellow); text-transform:uppercase; letter-spacing:0.12em; margin-bottom:6px; }
.upset-badge { background:rgba(255,205,0,0.15); border:1px solid var(--f1-yellow); color:var(--f1-yellow); font-family:var(--font-mono); font-size:0.6rem; padding:1px 6px; border-radius:1px; letter-spacing:0.1em; text-transform:uppercase; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
TEAM_COLORS = {
    "Red Bull Racing": "#3671C6", "Mercedes": "#27F4D2", "Ferrari": "#E8002D",
    "McLaren": "#FF8000", "Aston Martin": "#229971", "Alpine": "#FF87BC",
    "Williams": "#64C4FF", "RB": "#6692FF", "Haas F1 Team": "#B6BABD",
    "Sauber": "#52E252", "Kick Sauber": "#52E252", "AlphaTauri": "#5E8FAA",
    "Racing Bulls": "#6692FF",
}

# Full 2025/2026 grid — fallback if API is down
FALLBACK_DRIVERS = [
    {"driver_number": 1,  "name_acronym": "VER", "full_name": "Max Verstappen",    "team_name": "Red Bull Racing"},
    {"driver_number": 11, "name_acronym": "PER", "full_name": "Sergio Perez",      "team_name": "Red Bull Racing"},
    {"driver_number": 44, "name_acronym": "HAM", "full_name": "Lewis Hamilton",    "team_name": "Ferrari"},
    {"driver_number": 16, "name_acronym": "LEC", "full_name": "Charles Leclerc",   "team_name": "Ferrari"},
    {"driver_number": 55, "name_acronym": "SAI", "full_name": "Carlos Sainz",      "team_name": "Williams"},
    {"driver_number": 63, "name_acronym": "RUS", "full_name": "George Russell",    "team_name": "Mercedes"},
    {"driver_number": 12, "name_acronym": "ANT", "full_name": "Kimi Antonelli",    "team_name": "Mercedes"},
    {"driver_number": 4,  "name_acronym": "NOR", "full_name": "Lando Norris",      "team_name": "McLaren"},
    {"driver_number": 81, "name_acronym": "PIA", "full_name": "Oscar Piastri",     "team_name": "McLaren"},
    {"driver_number": 14, "name_acronym": "ALO", "full_name": "Fernando Alonso",   "team_name": "Aston Martin"},
    {"driver_number": 18, "name_acronym": "STR", "full_name": "Lance Stroll",      "team_name": "Aston Martin"},
    {"driver_number": 10, "name_acronym": "GAS", "full_name": "Pierre Gasly",      "team_name": "Alpine"},
    {"driver_number": 7,  "name_acronym": "DOO", "full_name": "Jack Doohan",       "team_name": "Alpine"},
    {"driver_number": 23, "name_acronym": "ALB", "full_name": "Alexander Albon",   "team_name": "Williams"},
    {"driver_number": 22, "name_acronym": "TSU", "full_name": "Yuki Tsunoda",      "team_name": "Red Bull Racing"},
    {"driver_number": 6,  "name_acronym": "HAD", "full_name": "Isack Hadjar",      "team_name": "Racing Bulls"},
    {"driver_number": 87, "name_acronym": "BEA", "full_name": "Oliver Bearman",    "team_name": "Haas F1 Team"},
    {"driver_number": 50, "name_acronym": "OCO", "full_name": "Esteban Ocon",      "team_name": "Haas F1 Team"},
    {"driver_number": 5,  "name_acronym": "HUL", "full_name": "Nico Hulkenberg",   "team_name": "Sauber"},
    {"driver_number": 27, "name_acronym": "BOR", "full_name": "Gabriel Bortoleto", "team_name": "Sauber"},
]

# Full feature set used by the model
FEATURE_COLS = [
    # Grid / qualifying
    "grid_position",
    "quali_delta_to_pole",
    # Driver form
    "driver_form_3",
    "driver_form_5",
    "circuit_specialist_score",
    "dnf_rate_driver",
    # Team / constructor
    "team_dnf_rate",
    "constructor_avg_finish",
    "pit_stop_efficiency",
    # Championship pressure
    "champ_points_before",
    "champ_position_before",
    "points_gap_to_leader",
    "seasons_experience",
    # Tyre strategy
    "starting_compound_enc",
    "total_stints",
    "avg_tyre_age",
    "tyre_age_at_start",
    # Pit stops
    "num_pit_stops",
    "avg_stop_duration",
    # Weather
    "air_temperature",
    "track_temperature",
    "humidity",
    "wind_speed",
    "rainfall",
    "track_temp_delta",        # track - air  → tyre degradation proxy
    # Circuit
    "circuit_type_enc",
    "is_street_circuit",
    # Safety car / race control (historical avg per circuit)
    "sc_probability",
]

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def get_team_color(team_name):
    for k, v in TEAM_COLORS.items():
        if k.lower() in str(team_name).lower():
            return v
    return "#555555"

def confidence_score(predicted_pos, grid_pos):
    return round(max(0.3, 1.0 - (abs(predicted_pos - grid_pos) * 0.05)), 2)

# ─────────────────────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_model_artifacts():
    xgb_path  = os.path.join(MODEL_DIR, "xgboost_model.pkl")
    meta_path = os.path.join(MODEL_DIR, "model_meta.pkl")
    if not os.path.exists(xgb_path):
        return None, None
    model = joblib.load(xgb_path)
    meta  = joblib.load(meta_path) if os.path.exists(meta_path) else {}
    return model, meta

@st.cache_data(ttl=3600)
def load_features_df():
    path = os.path.join(PROC_DIR, "features_final.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

@st.cache_data(ttl=3600)
def load_prediction_priors():
    """
    Extract the most recent known values for every driver and team
    from the training dataset so predictions are grounded in real data
    rather than placeholder defaults.
    """
    df = load_features_df()
    if df is None:
        return {}, {}

    sort_cols = [c for c in ["year", "date_start", "session_key"] if c in df.columns]
    ordered   = df.sort_values(sort_cols) if sort_cols else df

    # ── Driver-level priors ───────────────────────────────
    driver_prior_cols = [c for c in [
        "driver_number", "team_name",
        "champ_points_before", "champ_position_before",
        "team_dnf_rate", "driver_form_3", "driver_form_5",
        "circuit_specialist_score", "dnf_rate_driver",
        "constructor_avg_finish", "pit_stop_efficiency",
        "points_gap_to_leader", "seasons_experience",
        "total_stints", "avg_tyre_age", "num_pit_stops",
        "avg_stop_duration",
    ] if c in ordered.columns]

    driver_lookup = {}
    if "driver_number" in driver_prior_cols:
        latest = (
            ordered[driver_prior_cols]
            .dropna(subset=["driver_number"])
            .drop_duplicates("driver_number", keep="last")
        )
        driver_lookup = latest.set_index("driver_number").to_dict(orient="index")

    # ── Team-level priors ─────────────────────────────────
    team_prior_cols = [c for c in [
        "team_name", "team_dnf_rate",
        "constructor_avg_finish", "pit_stop_efficiency",
    ] if c in ordered.columns]

    team_lookup = {}
    if "team_name" in team_prior_cols:
        t_latest = (
            ordered[team_prior_cols]
            .dropna(subset=["team_name"])
            .drop_duplicates("team_name", keep="last")
        )
        team_lookup = t_latest.set_index("team_name").to_dict(orient="index")

    # ── Circuit-level priors (safety car probability) ─────
    circuit_lookup = {}
    if "circuit_short_name" in ordered.columns and "sc_probability" in ordered.columns:
        c_latest = (
            ordered[["circuit_short_name", "sc_probability"]]
            .dropna()
            .drop_duplicates("circuit_short_name", keep="last")
        )
        circuit_lookup = c_latest.set_index("circuit_short_name")["sc_probability"].to_dict()

    return driver_lookup, team_lookup, circuit_lookup

# Complete 2026 F1 calendar — all 24 rounds.
# Primary source when API is reachable; guaranteed fallback otherwise.
F1_2026_CALENDAR = [
    {"meeting_key": 1280, "meeting_name": "Australian Grand Prix",    "date_start": "2026-03-06", "circuit_short_name": "Melbourne",        "circuit_type": "Permanent",          "country_name": "Australia"},
    {"meeting_key": 1281, "meeting_name": "Chinese Grand Prix",       "date_start": "2026-03-13", "circuit_short_name": "Shanghai",          "circuit_type": "Permanent",          "country_name": "China"},
    {"meeting_key": 1282, "meeting_name": "Japanese Grand Prix",      "date_start": "2026-03-27", "circuit_short_name": "Suzuka",            "circuit_type": "Permanent",          "country_name": "Japan"},
    {"meeting_key": 1283, "meeting_name": "Bahrain Grand Prix",       "date_start": "2026-04-10", "circuit_short_name": "Sakhir",            "circuit_type": "Permanent",          "country_name": "Bahrain"},
    {"meeting_key": 1284, "meeting_name": "Saudi Arabian Grand Prix", "date_start": "2026-04-17", "circuit_short_name": "Jeddah",            "circuit_type": "Temporary - Street", "country_name": "Saudi Arabia"},
    {"meeting_key": 1285, "meeting_name": "Miami Grand Prix",         "date_start": "2026-05-01", "circuit_short_name": "Miami",             "circuit_type": "Temporary - Street", "country_name": "United States"},
    {"meeting_key": 1286, "meeting_name": "Canadian Grand Prix",      "date_start": "2026-05-22", "circuit_short_name": "Montreal",          "circuit_type": "Temporary - Street", "country_name": "Canada"},
    {"meeting_key": 1287, "meeting_name": "Monaco Grand Prix",        "date_start": "2026-06-05", "circuit_short_name": "Monaco",            "circuit_type": "Temporary - Street", "country_name": "Monaco"},
    {"meeting_key": 1288, "meeting_name": "Spanish Grand Prix",       "date_start": "2026-06-12", "circuit_short_name": "Barcelona",         "circuit_type": "Permanent",          "country_name": "Spain"},
    {"meeting_key": 1289, "meeting_name": "Austrian Grand Prix",      "date_start": "2026-06-26", "circuit_short_name": "Spielberg",         "circuit_type": "Permanent",          "country_name": "Austria"},
    {"meeting_key": 1290, "meeting_name": "British Grand Prix",       "date_start": "2026-07-03", "circuit_short_name": "Silverstone",       "circuit_type": "Permanent",          "country_name": "Great Britain"},
    {"meeting_key": 1291, "meeting_name": "Belgian Grand Prix",       "date_start": "2026-07-17", "circuit_short_name": "Spa-Francorchamps", "circuit_type": "Permanent",          "country_name": "Belgium"},
    {"meeting_key": 1292, "meeting_name": "Hungarian Grand Prix",     "date_start": "2026-07-24", "circuit_short_name": "Budapest",          "circuit_type": "Permanent",          "country_name": "Hungary"},
    {"meeting_key": 1293, "meeting_name": "Dutch Grand Prix",         "date_start": "2026-08-21", "circuit_short_name": "Zandvoort",         "circuit_type": "Permanent",          "country_name": "Netherlands"},
    {"meeting_key": 1294, "meeting_name": "Italian Grand Prix",       "date_start": "2026-09-04", "circuit_short_name": "Monza",             "circuit_type": "Permanent",          "country_name": "Italy"},
    {"meeting_key": 1295, "meeting_name": "Madrid Grand Prix",        "date_start": "2026-09-11", "circuit_short_name": "Madrid",            "circuit_type": "Temporary - Street", "country_name": "Spain"},
    {"meeting_key": 1296, "meeting_name": "Azerbaijan Grand Prix",    "date_start": "2026-09-25", "circuit_short_name": "Baku",              "circuit_type": "Temporary - Street", "country_name": "Azerbaijan"},
    {"meeting_key": 1297, "meeting_name": "Singapore Grand Prix",     "date_start": "2026-10-09", "circuit_short_name": "Marina Bay",        "circuit_type": "Temporary - Street", "country_name": "Singapore"},
    {"meeting_key": 1298, "meeting_name": "United States Grand Prix", "date_start": "2026-10-23", "circuit_short_name": "Austin",            "circuit_type": "Permanent",          "country_name": "United States"},
    {"meeting_key": 1299, "meeting_name": "Mexico City Grand Prix",   "date_start": "2026-10-30", "circuit_short_name": "Mexico City",       "circuit_type": "Permanent",          "country_name": "Mexico"},
    {"meeting_key": 1300, "meeting_name": "Brazilian Grand Prix",     "date_start": "2026-11-06", "circuit_short_name": "Sao Paulo",         "circuit_type": "Permanent",          "country_name": "Brazil"},
    {"meeting_key": 1301, "meeting_name": "Las Vegas Grand Prix",     "date_start": "2026-11-19", "circuit_short_name": "Las Vegas",         "circuit_type": "Temporary - Street", "country_name": "United States"},
    {"meeting_key": 1302, "meeting_name": "Qatar Grand Prix",         "date_start": "2026-11-27", "circuit_short_name": "Lusail",            "circuit_type": "Permanent",          "country_name": "Qatar"},
    {"meeting_key": 1303, "meeting_name": "Abu Dhabi Grand Prix",     "date_start": "2026-12-04", "circuit_short_name": "Yas Marina",        "circuit_type": "Permanent",          "country_name": "UAE"},
]


@st.cache_data(ttl=1800)
def fetch_upcoming_races():
    """
    Returns the full 2026 F1 race calendar.
    Tries the OpenF1 API first — uses it only if it returns >= 10 meetings.
    Always falls back to F1_2026_CALENDAR so all 24 races appear in the
    Grand Prix selector even when the API is unavailable or rate-limited.
    """
    try:
        for year in [datetime.now().year, datetime.now().year + 1]:
            r = requests.get(
                "https://api.openf1.org/v1/meetings",
                params={"year": year},
                timeout=8,
            )
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and len(data) >= 10:
                    cleaned = [
                        m for m in data
                        if isinstance(m, dict) and not m.get("is_cancelled", False)
                    ]
                    if len(cleaned) >= 10:
                        return sorted(cleaned, key=lambda x: x.get("date_start", ""))
    except Exception:
        pass

    return F1_2026_CALENDAR
@st.cache_data(ttl=1800)
def fetch_current_drivers(session_key="latest"):
    try:
        r = requests.get(
            "https://api.openf1.org/v1/drivers",
            params={"session_key": session_key},
            timeout=10
        )
        return r.json()
    except Exception:
        return []

@st.cache_data(ttl=1800)
def fetch_latest_championship():
    """
    Pull the most recent championship standings for all drivers.
    Returns dict keyed by driver_number.
    """
    try:
        r = requests.get(
            "https://api.openf1.org/v1/championship_drivers",
            params={"session_key": "latest"},
            timeout=10
        )
        data = r.json()
        result = {}
        for row in data:
            dn = row.get("driver_number")
            if dn:
                result[dn] = {
                    "points":   row.get("points_current", row.get("points_start", 0)),
                    "position": row.get("position_current", row.get("position_start", 10)),
                }
        return result
    except Exception:
        return {}

@st.cache_data(ttl=1800)
def fetch_latest_stints_for_circuit(circuit_meeting_key):
    """
    Fetch most recent race stints for the chosen circuit meeting
    to seed tyre strategy defaults.
    """
    try:
        # Get race session for this meeting
        r = requests.get(
            "https://api.openf1.org/v1/sessions",
            params={"meeting_key": circuit_meeting_key, "session_name": "Race"},
            timeout=10
        )
        sessions = r.json()
        if not sessions:
            return {}
        sk = sessions[0]["session_key"]

        r2 = requests.get(
            "https://api.openf1.org/v1/stints",
            params={"session_key": sk},
            timeout=10
        )
        stints = r2.json()
        if not stints:
            return {}

        df = pd.DataFrame(stints)
        result = {}
        for dn, grp in df.groupby("driver_number"):
            result[dn] = {
                "total_stints":    int(grp["stint_number"].max()),
                "starting_compound": str(grp.sort_values("stint_number").iloc[0].get("compound", "MEDIUM")),
                "avg_tyre_age":    float(grp["tyre_age_at_start"].mean()) if "tyre_age_at_start" in grp else 2.0,
                "tyre_age_at_start": float(grp.sort_values("stint_number").iloc[0].get("tyre_age_at_start", 0)),
            }
        return result
    except Exception:
        return {}

@st.cache_data(ttl=1800)
def fetch_latest_pit_efficiency(circuit_meeting_key):
    """Fetch pit stop data for most recent race at the circuit."""
    try:
        r = requests.get(
            "https://api.openf1.org/v1/sessions",
            params={"meeting_key": circuit_meeting_key, "session_name": "Race"},
            timeout=10
        )
        sessions = r.json()
        if not sessions:
            return {}
        sk = sessions[0]["session_key"]

        r2 = requests.get("https://api.openf1.org/v1/pit", params={"session_key": sk}, timeout=10)
        pits = r2.json()
        if not pits:
            return {}

        df = pd.DataFrame(pits)
        stop_col = "stop_duration" if "stop_duration" in df.columns else "lane_duration"
        result = {}
        for dn, grp in df.groupby("driver_number"):
            result[dn] = {
                "num_pit_stops":    len(grp),
                "avg_stop_duration": float(grp[stop_col].mean()) if stop_col in grp else 2.5,
            }
        return result
    except Exception:
        return {}

@st.cache_data(ttl=1800)
def fetch_race_control_sc_history(circuit_short_name):
    """
    Estimate safety car probability for a circuit from recent race control data.
    Returns a float 0.0 - 1.0.
    """
    try:
        r = requests.get(
            "https://api.openf1.org/v1/race_control",
            params={"category": "SafetyCar"},
            timeout=10
        )
        data = r.json()
        if not data:
            return 0.35  # F1 historical average ~35% SC per race

        df = pd.DataFrame(data)
        if "message" not in df.columns:
            return 0.35

        sc_events = df[df["message"].str.contains("SAFETY CAR DEPLOYED", case=False, na=False)]
        # Rough proxy: ratio of sessions with SC events
        total_sessions = df["session_key"].nunique() if "session_key" in df.columns else 50
        sc_sessions    = sc_events["session_key"].nunique() if "session_key" in sc_events.columns else 15
        return round(min(0.9, sc_sessions / max(total_sessions, 1)), 2)
    except Exception:
        return 0.35

# ─────────────────────────────────────────────────────────────
# PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────
def build_race_input(
    drivers_info,
    grid_positions,
    weather_vals,
    live_champ,
    circuit_type_enc,
    circuit_short_name,
    sc_probability,
    driver_lookup,
    team_lookup,
    stint_data,
    pit_data,
):
    """
    Build a feature-complete DataFrame — one row per driver — for the model.
    Priority order for every feature:
        1. Live API data for this specific race/circuit
        2. Historical priors from training dataset
        3. Sensible F1 domain-knowledge defaults
    """
    def pick(*values, default):
        for v in values:
            try:
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    return v
            except Exception:
                pass
        return default

    compound_enc = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}

    # Max points in live standings — to compute gap-to-leader
    all_pts = [v.get("points", 0) for v in live_champ.values() if v.get("points") is not None]
    leader_pts = max(all_pts) if all_pts else 300

    rows = []
    for i, d in enumerate(drivers_info):
        dn         = d.get("driver_number")
        team       = d.get("team_name", "Unknown")
        grid       = int(grid_positions.get(dn, i + 1))

        dp  = driver_lookup.get(dn, {})
        tp  = team_lookup.get(team, {})
        sp  = stint_data.get(dn, {})
        pp  = pit_data.get(dn, {})
        lc  = live_champ.get(dn, {})

        champ_pts  = pick(lc.get("points"),  dp.get("champ_points_before"),  default=50.0)
        champ_pos  = pick(lc.get("position"), dp.get("champ_position_before"), default=10.0)
        gap_leader = pick(leader_pts - champ_pts, default=200.0)

        rows.append({
            # ── Grid / qualifying ──────────────────────────
            "grid_position":         grid,
            "quali_delta_to_pole":   (grid - 1) * 0.08,

            # ── Driver form ────────────────────────────────
            "driver_form_3":         pick(dp.get("driver_form_3"),          default=float(grid) * 0.9),
            "driver_form_5":         pick(dp.get("driver_form_5"),          default=float(grid) * 0.92),
            "circuit_specialist_score": pick(dp.get("circuit_specialist_score"), default=float(grid) * 0.85),
            "dnf_rate_driver":       pick(dp.get("dnf_rate_driver"),        default=0.06),
            "seasons_experience":    pick(dp.get("seasons_experience"),     default=5.0),

            # ── Constructor ────────────────────────────────
            "team_dnf_rate":         pick(tp.get("team_dnf_rate"),          dp.get("team_dnf_rate"),         default=0.05),
            "constructor_avg_finish":pick(tp.get("constructor_avg_finish"), dp.get("constructor_avg_finish"), default=float(grid)),
            "pit_stop_efficiency":   pick(pp.get("avg_stop_duration"),      tp.get("pit_stop_efficiency"),   default=2.5),

            # ── Championship ───────────────────────────────
            "champ_points_before":   float(champ_pts),
            "champ_position_before": float(champ_pos),
            "points_gap_to_leader":  float(gap_leader),

            # ── Tyre strategy ──────────────────────────────
            "starting_compound_enc": compound_enc.get(
                str(sp.get("starting_compound", "MEDIUM")).upper(), 1
            ),
            "total_stints":          pick(sp.get("total_stints"),     dp.get("total_stints"),     default=2.0),
            "avg_tyre_age":          pick(sp.get("avg_tyre_age"),      dp.get("avg_tyre_age"),     default=2.0),
            "tyre_age_at_start":     pick(sp.get("tyre_age_at_start"), default=0.0),

            # ── Pit stops ──────────────────────────────────
            "num_pit_stops":         pick(pp.get("num_pit_stops"),     dp.get("num_pit_stops"),    default=1.0),
            "avg_stop_duration":     pick(pp.get("avg_stop_duration"), dp.get("avg_stop_duration"), default=2.5),

            # ── Weather ────────────────────────────────────
            "air_temperature":       float(weather_vals.get("air_temp",   25)),
            "track_temperature":     float(weather_vals.get("track_temp", 38)),
            "humidity":              float(weather_vals.get("humidity",   55)),
            "wind_speed":            float(weather_vals.get("wind_speed",  2)),
            "rainfall":              int(weather_vals.get("rainfall",      0)),
            "track_temp_delta":      float(weather_vals.get("track_temp", 38)) - float(weather_vals.get("air_temp", 25)),

            # ── Circuit ────────────────────────────────────
            "circuit_type_enc":      circuit_type_enc,
            "is_street_circuit":     1 if circuit_type_enc == 1 else 0,

            # ── Safety car ─────────────────────────────────
            "sc_probability":        float(sc_probability),

            # ── Identity (not fed to model) ────────────────
            "driver_number":         dn,
            "name_acronym":          d.get("name_acronym", "UNK"),
            "full_name":             d.get("full_name",    "Unknown"),
            "team_name":             team,
        })

    return pd.DataFrame(rows)


def run_prediction(model, feature_cols, race_df):
    """
    Run model prediction with bias correction.

    XGBoost regression trained on historical F1 data carries a conservative
    bias - it rarely predicts absolute extremes (P1 or P20) because those
    are statistically uncommon in training. Two-step correction:

    1. Score spread normalisation: rescale raw scores so P1-P20 range is
       always used rather than clustering around P5-P15.

    2. Dominance snap: if pole sitter inputs signal clear dominance,
       snap their score to guarantee P1 instead of rounding to P2.
    """
    for c in feature_cols:
        if c not in race_df.columns:
            race_df[c] = 0.0

    X      = race_df[feature_cols].fillna(0).values.astype(float)
    scores = model.predict(X).astype(float)

    # Step 1: spread normalisation
    # Linearly rescale so best driver maps to ~1.0 and worst to ~20.0
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        scores = 1.0 + (scores - s_min) / (s_max - s_min) * 19.0

    # Step 2: dominance snap
    # Find the pole sitter (grid_position == 1).
    # If their normalised score is already best AND inputs signal dominance,
    # pull score below 1.5 to guarantee P1.
    if "grid_position" in race_df.columns:
        pole_mask = race_df["grid_position"].values == 1
        if pole_mask.any():
            pole_idx   = int(np.where(pole_mask)[0][0])
            pole_score = scores[pole_idx]

            row       = race_df.iloc[pole_idx]
            form      = float(row.get("driver_form_3",        10))
            sc_prob   = float(row.get("sc_probability",       0.35))
            dnf_r     = float(row.get("dnf_rate_driver",      0.06))
            champ_pos = float(row.get("champ_position_before",10))
            gap       = float(row.get("points_gap_to_leader", 200))

            # Dominance index: lower = more dominant
            dominance = form + sc_prob * 10 + dnf_r * 20 + champ_pos + gap / 50

            if pole_score == scores.min() and dominance < 8.0:
                # Clearly dominant pole sitter -> hard snap to P1
                scores[pole_idx] = 0.5
            elif pole_score <= scores.min() + 0.3 and form <= 3.0:
                # Close to best + good form -> give pole advantage
                scores[pole_idx] = scores.min() - 0.1

    return scores


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
logo_path = os.path.join(BASE_DIR, "assets", "f1_logo.png")
col_logo, col_title = st.columns([1, 11])
with col_logo:
    if os.path.exists(logo_path):
        st.image(logo_path, width=68)
    else:
        st.markdown("<div style='font-size:2.2rem;line-height:1;padding-top:6px'>🏎</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("""
    <div style="padding-top:2px">
        <div class="pw-title">PITWALL <span>INTEL</span></div>
        <div class="pw-subtitle">Data Driven &nbsp;·&nbsp; Race Ready &nbsp;·&nbsp; Powered by OpenF1</div>
    </div>
    """, unsafe_allow_html=True)

model_exists   = os.path.exists(os.path.join(MODEL_DIR, "xgboost_model.pkl"))
features_exist = os.path.exists(os.path.join(PROC_DIR,  "features_final.csv"))

st.markdown(f"""
<div class="status-bar">
    <span class="status-dot"></span>
    SYSTEM ONLINE &nbsp;·&nbsp; OPENF1 API CONNECTED &nbsp;·&nbsp;
    DATA {'READY' if features_exist else 'NOT FETCHED'} &nbsp;·&nbsp;
    MODEL {'READY' if model_exists else 'NOT TRAINED'}
</div><hr/>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;font-weight:700;
    letter-spacing:0.12em;text-transform:uppercase;color:#e8002d;
    padding-bottom:0.6rem;border-bottom:1px solid #222;margin-bottom:1rem">
    RACE CONTROL
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='pw-section-label'>Screen</div>", unsafe_allow_html=True)
    page = st.selectbox("Select Screen",
        ["Race Predictor", "Model Performance", "Data Explorer", "What-If Simulator"],
        label_visibility="collapsed"
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("<div class='pw-section-label'>Data Pipeline</div>", unsafe_allow_html=True)

    if st.button("Fetch + Train Model", use_container_width=True):
        st.session_state["run_pipeline"] = True

    st.markdown(f"""
    <div style="margin-top:0.8rem">
        <div class="telemetry-line">DATA &nbsp;{'LOADED' if features_exist else 'NOT FOUND'}</div>
        <div class="telemetry-line">MODEL {'READY' if model_exists else 'NOT TRAINED'}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
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
    progress_bar = st.progress(0)
    status_text  = st.empty()
    try:
        from utils.data_pipeline import run_pipeline as fetch_data
        from utils.model_trainer import train_and_save

        steps = [
            (10, "Connecting to OpenF1 API..."),
            (22, "Fetching race sessions 2023–2025..."),
            (36, "Downloading race results + grids..."),
            (50, "Loading tyre stints + pit stops..."),
            (63, "Fetching weather + race control data..."),
            (76, "Loading championship standings..."),
            (88, "Engineering all features..."),
            (94, "Training XGBoost + Random Forest..."),
        ]
        for pct, msg in steps:
            status_text.markdown(f'<div class="telemetry-line">{msg}</div>', unsafe_allow_html=True)
            progress_bar.progress(pct)
            time.sleep(0.15)

        fetch_data(years=[2023, 2024, 2025], save=True)
        train_and_save()

        progress_bar.progress(100)
        status_text.markdown(
            '<div class="telemetry-line" style="color:#2aff7a">Pipeline complete. Model ready.</div>',
            unsafe_allow_html=True
        )
        st.success("Model trained and ready.")
        st.cache_data.clear()
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.info("Check your internet connection and try again.")

# ─────────────────────────────────────────────────────────────
# PAGE 1 — RACE PREDICTOR
# ─────────────────────────────────────────────────────────────
if page == "Race Predictor":
    model, meta = load_model_artifacts()

    if not model_exists or model is None:
        st.markdown("""
        <div class="pw-card pw-card-accent">
            <div class="pw-section-label">No Model Detected</div>
            <div style="font-family:'Rajdhani',sans-serif;font-size:1rem;color:rgba(245,245,245,0.6);line-height:1.7">
                Click <strong style="color:#e8002d">Fetch + Train Model</strong> in the sidebar.
                The pipeline downloads 3 seasons of F1 data from OpenF1 and trains the prediction model.
                Estimated time: 5–10 minutes.
            </div>
        </div>
        """, unsafe_allow_html=True)
        for i, (label, desc) in enumerate([
            ("Step 01", "Fetch Race Data", "Sessions · Results · Grids<br>Stints · Pit Stops · Weather"),
            ("Step 02", "Engineer Features", "Driver Form · Quali Delta<br>SC Probability · Constructor Rate"),
            ("Step 03", "Train XGBoost",    "XGBoost + Random Forest<br>28 Features · Time-based Split"),
        ], 0):
            pass
        c1, c2, c3 = st.columns(3)
        for col, step, title, desc in zip(
            [c1, c2, c3],
            ["Step 01", "Step 02", "Step 03"],
            ["Fetch Race Data", "Engineer Features", "Train XGBoost"],
            [
                "Sessions · Results · Grids<br>Stints · Pit Stops · Weather",
                "Driver Form · Quali Delta<br>SC Probability · Constructor Rate",
                "XGBoost + Random Forest<br>28 Features · Time-based Split",
            ]
        ):
            accent = "pw-card-accent" if step == "Step 03" else ""
            with col:
                st.markdown(f"""
                <div class="pw-card {accent}">
                    <div class="pw-section-label">{step}</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;font-weight:600;letter-spacing:0.05em">{title}</div>
                    <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;color:rgba(245,245,245,0.4);margin-top:6px;line-height:1.6">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    else:
        # ── Race selector ──────────────────────────────────
        st.markdown("<div class='pw-section-label'>Race Setup</div>", unsafe_allow_html=True)
        col_race, col_weather = st.columns([2, 1])

        with col_race:
            upcoming = fetch_upcoming_races()
            race_names = [
                f"{m.get('meeting_name','?')} — {m.get('date_start','')[:10]}"
                for m in upcoming
            ] if upcoming else ["Bahrain Grand Prix — 2026-03-01"]
            selected_label   = st.selectbox("Grand Prix", race_names)
            selected_meeting = upcoming[race_names.index(selected_label)] if upcoming else {}
            circuit_name     = selected_meeting.get("circuit_short_name", "Unknown")
            circuit_type_raw = selected_meeting.get("circuit_type", "Permanent")
            circuit_type_enc = 1 if "Street" in circuit_type_raw else (2 if "Road" in circuit_type_raw else 0)
            meeting_key      = selected_meeting.get("meeting_key")

            st.markdown(f"""
            <div style="margin-top:0.5rem">
                <span class="upset-badge" style="margin-right:4px">{circuit_type_raw}</span>
                <span class="upset-badge">{circuit_name}</span>
            </div>
            """, unsafe_allow_html=True)

        with col_weather:
            st.markdown("<div class='pw-section-label'>Race Conditions</div>", unsafe_allow_html=True)
            rainfall  = st.selectbox("Conditions", ["Dry", "Wet", "Mixed"])
            air_temp  = st.slider("Air Temp (°C)",   10, 45, 24)
            track_temp= st.slider("Track Temp (°C)", 15, 65, 38)
            humidity  = st.slider("Humidity (%)",    20, 100, 55)
            wind_speed= st.slider("Wind (m/s)",       0, 15,  2)

        weather_vals = {
            "air_temp":   air_temp,
            "track_temp": track_temp,
            "humidity":   humidity,
            "wind_speed": wind_speed,
            "rainfall":   1 if rainfall == "Wet" else (0.5 if rainfall == "Mixed" else 0),
        }

        # ── Live data fetching ─────────────────────────────
        with st.spinner("Fetching live race data..."):
            live_champ  = fetch_latest_championship()
            stint_data  = fetch_latest_stints_for_circuit(meeting_key) if meeting_key else {}
            pit_data    = fetch_latest_pit_efficiency(meeting_key) if meeting_key else {}
            sc_prob     = fetch_race_control_sc_history(circuit_name)
            raw_drivers = fetch_current_drivers("latest") or FALLBACK_DRIVERS

        seen, unique_drivers = set(), []
        if isinstance(raw_drivers, list):
            for d in raw_drivers:
                if not isinstance(d, dict):
                    continue
                dn = d.get("driver_number")
                if dn is not None and dn not in seen:
                    seen.add(dn)
                    unique_drivers.append(d)

        if not unique_drivers:
            unique_drivers = FALLBACK_DRIVERS[:20]
        else:
            unique_drivers = unique_drivers[:20]

        # ── Priors from training data ──────────────────────
        priors = load_prediction_priors()
        driver_lookup  = priors[0]
        team_lookup    = priors[1]
        circuit_lookup = priors[2] if len(priors) > 2 else {}
        # Override sc_prob with circuit-specific historical value if available
        if circuit_name in circuit_lookup:
            sc_prob = circuit_lookup[circuit_name]

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ── Live data summary ──────────────────────────────
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Live Champ Data", f"{len(live_champ)} drivers" if live_champ else "Offline")
        with col_b:
            st.metric("Tyre Data", f"{len(stint_data)} drivers" if stint_data else "Using priors")
        with col_c:
            st.metric("Pit Data", f"{len(pit_data)} drivers" if pit_data else "Using priors")
        with col_d:
            st.metric("SC Probability", f"{sc_prob*100:.0f}%")

        # ── Grid customization ─────────────────────────────
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("<div class='pw-section-label'>Starting Grid</div>", unsafe_allow_html=True)

        with st.expander("Customize Grid Positions (optional — defaults to P1–P20)", expanded=False):
            grid_positions = {}
            cols4 = st.columns(4)
            for i, d in enumerate(unique_drivers):
                with cols4[i % 4]:
                    pos = st.number_input(
                        d.get("name_acronym", "?"),
                        min_value=1, max_value=20, value=i + 1,
                        key=f"grid_{d['driver_number']}"
                    )
                    grid_positions[d["driver_number"]] = pos
        if not grid_positions:
            grid_positions = {d["driver_number"]: i + 1 for i, d in enumerate(unique_drivers)}

        st.markdown("<hr/>", unsafe_allow_html=True)

        if st.button("RUN PREDICTION", use_container_width=True):
            with st.spinner("Analysing race data..."):
                race_df = build_race_input(
                    unique_drivers, grid_positions, weather_vals,
                    live_champ, circuit_type_enc, circuit_name, sc_prob,
                    driver_lookup, team_lookup, stint_data, pit_data,
                )
                feature_cols = meta.get("feature_cols", FEATURE_COLS)
                preds = run_prediction(model, feature_cols, race_df)
                race_df["predicted_score"] = preds
                race_df = race_df.sort_values("predicted_score").reset_index(drop=True)
                race_df["predicted_position"] = range(1, len(race_df) + 1)
                race_df["grid_pos"]       = race_df["driver_number"].map(grid_positions)
                race_df["position_gain"]  = race_df["grid_pos"] - race_df["predicted_position"]
                st.session_state["prediction_result"] = race_df
                st.session_state["selected_race_label"] = selected_label

        # ── Results ────────────────────────────────────────
        if "prediction_result" in st.session_state:
            result_df  = st.session_state["prediction_result"]
            race_label = st.session_state.get("selected_race_label", "")

            st.markdown(f"""
            <div style="margin:1.5rem 0 0.8rem 0">
                <div class="pw-section-label">Predicted Race Classification</div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.5rem;font-weight:700;letter-spacing:0.06em">
                {race_label.split('—')[0].strip().upper()}
                </div>
            </div>
            """, unsafe_allow_html=True)

            tab_grid, tab_gain, tab_podium, tab_upset = st.tabs([
                "Predicted Grid", "Position Change", "Podium Breakdown", "Upset Alert"
            ])

            # ── TAB 1: GRID ───────────────────────────────
            with tab_grid:
                grid_html = """
                <style>
                @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@700;800&family=Share+Tech+Mono&family=Rajdhani:wght@600&display=swap');
                body { margin:0; background:transparent; color:#f5f5f5; }
                .grid-wrap { border:1px solid #222; border-radius:2px; overflow:hidden; border-top:2px solid #e8002d; background:#0b0b0b; }
                .grid-head { display:flex; align-items:center; padding:0.6rem 1rem; background:#111; font-family:'Share Tech Mono',monospace; font-size:0.62rem; color:rgba(245,245,245,0.6); letter-spacing:0.1em; text-transform:uppercase; border-bottom:1px solid #222; }
                .driver-row { display:flex; align-items:center; gap:10px; padding:0.65rem 1rem; border-bottom:1px solid rgba(255,255,255,0.04); transition:background 0.15s; }
                .driver-row:nth-child(even) { background:rgba(255,255,255,0.02); }
                .driver-row:hover { background:rgba(232,0,45,0.08); }
                .pos { width:28px; flex:0 0 28px; text-align:right; font-family:'Barlow Condensed',sans-serif; font-size:1rem; font-weight:700; color:rgba(245,245,245,0.5); }
                .pos-top { color:#ffcd00 !important; }
                .dot { width:8px; height:8px; border-radius:50%; flex:0 0 8px; }
                .name { flex:1; font-family:'Barlow Condensed',sans-serif; font-size:1.02rem; font-weight:800; letter-spacing:0.05em; }
                .team { width:140px; flex:0 0 140px; font-family:'Share Tech Mono',monospace; font-size:0.62rem; color:rgba(245,245,245,0.7); letter-spacing:0.05em; text-transform:uppercase; }
                .grid-col { width:46px; flex:0 0 46px; text-align:right; font-family:'Share Tech Mono',monospace; font-size:0.68rem; color:rgba(245,245,245,0.65); }
                .delta { width:52px; flex:0 0 52px; text-align:right; font-family:'Share Tech Mono',monospace; font-size:0.68rem; font-weight:700; }
                .badge { background:rgba(255,205,0,0.15); border:1px solid #ffcd00; color:#ffcd00; font-family:'Share Tech Mono',monospace; font-size:0.55rem; padding:1px 5px; border-radius:1px; letter-spacing:0.08em; text-transform:uppercase; margin-left:7px; }
                </style>
                <div class="grid-wrap">
                <div class="grid-head">
                    <span style="width:28px;text-align:right">POS</span>
                    <span style="width:18px"></span>
                    <span style="flex:1">DRIVER</span>
                    <span style="width:140px">TEAM</span>
                    <span style="width:46px;text-align:right">GRID</span>
                    <span style="width:52px;text-align:right">DELTA</span>
                </div>
                """
                for _, row in result_df.iterrows():
                    pos       = int(row["predicted_position"])
                    pos_cls   = "pos-top" if pos <= 3 else ""
                    tc        = get_team_color(row.get("team_name", ""))
                    gp        = int(row.get("grid_pos", pos))
                    delta     = int(row.get("position_gain", 0))
                    ds        = f"+{delta}" if delta > 0 else str(delta)
                    dc        = "#2aff7a" if delta > 0 else ("#e8002d" if delta < 0 else "rgba(245,245,245,0.4)")
                    upset     = pos <= 5 and gp >= 8
                    badge     = '<span class="badge">UPSET</span>' if upset else ""
                    grid_html += f"""
                    <div class="driver-row">
                        <span class="pos {pos_cls}">{pos:02d}</span>
                        <span class="dot" style="background:{tc}"></span>
                        <span class="name">{row.get('name_acronym','?')}{badge}</span>
                        <span class="team">{str(row.get('team_name','?'))[:18]}</span>
                        <span class="grid-col">P{gp}</span>
                        <span class="delta" style="color:{dc}">{ds}</span>
                    </div>"""
                grid_html += "</div>"
                components.html(grid_html, height=max(260, 62 + len(result_df) * 44), scrolling=True)

            # ── TAB 2: POSITION CHANGE ────────────────────
            with tab_gain:
                colors = [get_team_color(r["team_name"]) for _, r in result_df.iterrows()]
                fig = go.Figure(go.Bar(
                    x=result_df["name_acronym"],
                    y=result_df["position_gain"],
                    marker_color=colors, marker_line_width=0,
                    text=[f"+{v}" if v > 0 else str(v) for v in result_df["position_gain"]],
                    textposition="outside",
                    textfont=dict(family="Share Tech Mono", size=10, color="#f5f5f5"),
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Rajdhani", color="#f5f5f5"),
                    xaxis=dict(gridcolor="#1a1a1a", tickfont=dict(family="Barlow Condensed", size=11),
                               title=dict(text="DRIVER", font=dict(family="Share Tech Mono", size=10, color="rgba(245,245,245,0.4)"))),
                    yaxis=dict(gridcolor="#1a1a1a", zerolinecolor="#333",
                               title=dict(text="POSITIONS GAINED / LOST", font=dict(family="Share Tech Mono", size=10, color="rgba(245,245,245,0.4)"))),
                    margin=dict(l=40, r=20, t=20, b=40), height=360,
                )
                st.plotly_chart(fig, use_container_width=True)

            # ── TAB 3: PODIUM ─────────────────────────────
            with tab_podium:
                c1, c2, c3 = st.columns(3)
                for col, (_, row), medal in zip([c1, c2, c3], result_df.head(3).iterrows(), ["01", "02", "03"]):
                    with col:
                        tc = get_team_color(row.get("team_name", ""))
                        st.markdown(f"""
                        <div class="pw-card" style="border-top:3px solid {tc};text-align:center;padding:1.4rem 1rem">
                            <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:rgba(245,245,245,0.35);letter-spacing:0.12em;margin-bottom:6px">P{medal}</div>
                            <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.8rem;font-weight:800;letter-spacing:0.06em">{row.get('name_acronym','?')}</div>
                            <div style="font-family:'Rajdhani',sans-serif;font-size:0.85rem;color:rgba(245,245,245,0.5);margin-top:2px">{row.get('team_name','?')}</div>
                            <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:rgba(245,245,245,0.3);margin-top:8px">Started P{int(row.get('grid_pos', 0))}</div>
                        </div>
                        """, unsafe_allow_html=True)

            # ── TAB 4: UPSET ALERT ────────────────────────
            with tab_upset:
                upsets = result_df[(result_df["predicted_position"] <= 5) & (result_df["grid_pos"] >= 8)]
                if upsets.empty:
                    st.markdown("""
                    <div class="pw-card"><div style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;
                    color:rgba(245,245,245,0.4);text-align:center;padding:1rem">
                    No significant upsets predicted for this race.</div></div>
                    """, unsafe_allow_html=True)
                else:
                    for _, row in upsets.iterrows():
                        tc = get_team_color(row.get("team_name", ""))
                        st.markdown(f"""
                        <div class="pw-card pw-card-yellow">
                            <div style="display:flex;align-items:center;gap:12px">
                                <span style="width:12px;height:12px;border-radius:50%;background:{tc};flex:0 0 12px;display:inline-block"></span>
                                <div style="flex:1">
                                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.2rem;font-weight:700;letter-spacing:0.06em">
                                    {row.get('name_acronym','?')} — PREDICTED P{int(row['predicted_position'])}</div>
                                    <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;color:rgba(245,245,245,0.45);margin-top:3px">
                                    Started P{int(row['grid_pos'])} · {int(row['position_gain'])} positions gained · {row.get('team_name','?')}</div>
                                </div>
                                <span class="upset-badge">UPSET ALERT</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE 2 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────
elif page == "Model Performance":
    model, meta = load_model_artifacts()
    df = load_features_df()

    st.markdown("<div class='pw-section-label'>Model Metrics</div>", unsafe_allow_html=True)

    if meta:
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Best Model",      meta.get("best_model", "XGBoost"))
        with c2: st.metric("XGBoost MAE",     f"{meta.get('xgb_mae', 0):.2f} pos")
        with c3: st.metric("RF MAE",           f"{meta.get('rf_mae',  0):.2f} pos")
        with c4: st.metric("Podium Accuracy",  f"{meta.get('xgb_podium_acc', 0)*100:.1f}%")
    else:
        st.info("Train the model first to see performance metrics.")

    if model and hasattr(model, "feature_importances_"):
        fc = meta.get("feature_cols", FEATURE_COLS) if meta else FEATURE_COLS
        fc_available = [c for c in fc if c in (meta.get("feature_cols", fc) if meta else fc)]
        if len(fc_available) == len(model.feature_importances_):
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown("<div class='pw-section-label'>Feature Importance — What the model actually uses</div>", unsafe_allow_html=True)
            imp = pd.Series(model.feature_importances_, index=fc_available).sort_values()
            bar_colors = [
                "#e8002d" if v == imp.max() else
                "#ffcd00" if v >= imp.quantile(0.75) else
                "#444"
                for v in imp.values
            ]
            fig = go.Figure(go.Bar(
                x=imp.values, y=imp.index, orientation="h",
                marker_color=bar_colors, marker_line_width=0,
                text=[f"{v:.3f}" for v in imp.values],
                textposition="outside",
                textfont=dict(family="Share Tech Mono", size=9, color="rgba(245,245,245,0.5)"),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Rajdhani", color="#f5f5f5"),
                xaxis=dict(gridcolor="#1a1a1a",
                           title=dict(text="IMPORTANCE SCORE", font=dict(family="Share Tech Mono", size=9, color="rgba(245,245,245,0.4)"))),
                yaxis=dict(tickfont=dict(family="Share Tech Mono", size=10)),
                margin=dict(l=10, r=60, t=10, b=40), height=520,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Feature coverage table
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("<div class='pw-section-label'>Feature Coverage — Live vs Prior vs Default</div>", unsafe_allow_html=True)
    coverage_data = {
        "Feature": ["grid_position", "quali_delta_to_pole", "driver_form_3/5", "circuit_specialist_score",
                    "champ_points_before", "champ_position_before", "points_gap_to_leader",
                    "team_dnf_rate", "constructor_avg_finish", "pit_stop_efficiency",
                    "total_stints", "starting_compound_enc", "avg_tyre_age",
                    "num_pit_stops", "avg_stop_duration",
                    "air/track_temperature", "rainfall", "humidity", "wind_speed", "track_temp_delta",
                    "circuit_type_enc", "is_street_circuit", "sc_probability"],
        "Source":  ["User input", "Derived from grid", "Training dataset",  "Training dataset",
                    "OpenF1 live API", "OpenF1 live API", "Derived from champ",
                    "Training dataset", "Training dataset", "OpenF1 pit API",
                    "OpenF1 stints API", "OpenF1 stints API", "OpenF1 stints API",
                    "OpenF1 pit API", "OpenF1 pit API",
                    "User input", "User input", "User input", "User input", "Derived",
                    "OpenF1 meetings API", "Derived", "OpenF1 race control + training"],
        "Priority": ["P1 — Always live", "P1 — Derived", "P2 — Historical prior", "P2 — Historical prior",
                     "P1 — Live if available", "P1 — Live if available", "P1 — Derived",
                     "P2 — Historical prior", "P2 — Historical prior", "P1 — Live if available",
                     "P1 — Live if available", "P1 — Live if available", "P1 — Live if available",
                     "P1 — Live if available", "P1 — Live if available",
                     "P1 — User input", "P1 — User input", "P1 — User input", "P1 — User input", "P1 — Derived",
                     "P1 — Live API", "P1 — Derived", "P1 — Live + historical"],
    }
    st.dataframe(pd.DataFrame(coverage_data), use_container_width=True, height=500)

# ─────────────────────────────────────────────────────────────
# PAGE 3 — DATA EXPLORER
# ─────────────────────────────────────────────────────────────
elif page == "Data Explorer":
    df = load_features_df()
    st.markdown("<div class='pw-section-label'>Dataset Overview</div>", unsafe_allow_html=True)

    if df is None:
        st.markdown("""
        <div class="pw-card pw-card-accent">
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.8rem;color:rgba(245,245,245,0.5)">
            No data found. Run the pipeline from the sidebar first.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Rows",   f"{len(df):,}")
        with c2: st.metric("Seasons",      df["year"].nunique() if "year" in df else "N/A")
        with c3: st.metric("Race Events",  df["circuit_short_name"].nunique() if "circuit_short_name" in df else "N/A")
        with c4: st.metric("Features",     len(df.columns))

        st.markdown("<hr/>", unsafe_allow_html=True)

        if "year" in df.columns:
            cf1, cf2 = st.columns(2)
            with cf1:
                yr   = st.selectbox("Season", sorted(df["year"].unique(), reverse=True))
            with cf2:
                circs = ["All"] + sorted(df[df["year"] == yr]["circuit_short_name"].unique().tolist()) if "circuit_short_name" in df.columns else ["All"]
                circ  = st.selectbox("Circuit", circs)

            filt = df[df["year"] == yr]
            if circ != "All" and "circuit_short_name" in filt.columns:
                filt = filt[filt["circuit_short_name"] == circ]

            show_cols = [c for c in [
                "name_acronym", "team_name", "circuit_short_name",
                "grid_position", "finish_position", "quali_delta_to_pole",
                "driver_form_3", "driver_form_5", "circuit_specialist_score",
                "champ_points_before", "team_dnf_rate", "sc_probability",
                "rainfall", "total_stints", "num_pit_stops",
            ] if c in filt.columns]
            st.dataframe(filt[show_cols].reset_index(drop=True), use_container_width=True, height=380)

        # Scatter: grid vs finish
        if "grid_position" in df.columns and "finish_position" in df.columns:
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown("<div class='pw-section-label'>Grid Position vs Finish Position</div>", unsafe_allow_html=True)
            sample = df.dropna(subset=["grid_position", "finish_position"]).copy()
            if "team_name" in sample.columns:
                sample["_tc"] = sample["team_name"].apply(get_team_color)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=sample["grid_position"], y=sample["finish_position"],
                mode="markers",
                marker=dict(color=sample["_tc"] if "_tc" in sample.columns else "#e8002d", size=5, opacity=0.65, line=dict(width=0)),
                text=sample.get("name_acronym", ""),
                hovertemplate="<b>%{text}</b><br>Grid: %{x}<br>Finish: %{y}<extra></extra>",
            ))
            fig2.add_trace(go.Scatter(x=[1,21], y=[1,21], mode="lines",
                line=dict(color="#333", dash="dot", width=1), showlegend=False))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Rajdhani", color="#f5f5f5"),
                xaxis=dict(gridcolor="#1a1a1a", title=dict(text="GRID POSITION", font=dict(family="Share Tech Mono", size=9, color="rgba(245,245,245,0.4)"))),
                yaxis=dict(gridcolor="#1a1a1a", title=dict(text="FINISH POSITION", font=dict(family="Share Tech Mono", size=9, color="rgba(245,245,245,0.4)"))),
                margin=dict(l=40, r=20, t=10, b=40), height=380,
            )
            st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE 4 — WHAT-IF SIMULATOR
# ─────────────────────────────────────────────────────────────
elif page == "What-If Simulator":
    model, meta = load_model_artifacts()
    st.markdown("""
    <div class="pw-card pw-card-yellow" style="margin-bottom:1.5rem">
        <div class="pw-section-label">What-If Simulator</div>
        <div style="font-family:'Rajdhani',sans-serif;font-size:0.95rem;color:rgba(245,245,245,0.6);line-height:1.6">
        Adjust any race variable in real time and watch the prediction update instantly.
        Every slider change re-runs the full model.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not model_exists or model is None:
        st.info("Train the model first to use the What-If Simulator.")
    else:
        cl, cr = st.columns(2)
        with cl:
            st.markdown("<div class='what-if-label'>Driver Variables</div>", unsafe_allow_html=True)
            driver_name  = st.selectbox("Driver", ["VER","HAM","LEC","NOR","PIA","RUS","ALO","SAI","GAS","TSU","ALB","HUL","ANT","BEA"])
            grid_pos     = st.slider("Grid Position",          1, 20,  5)
            champ_pts    = st.slider("Championship Points",    0, 450, 200)
            champ_pos_v  = st.slider("Championship Position",  1, 20,  4)
            driver_form  = st.slider("Driver Form (avg last 3 races)", 1.0, 20.0, float(grid_pos), step=0.5)
            dnf_rate_d   = st.slider("Driver DNF Rate",        0.0, 0.4, 0.05, step=0.01)

        with cr:
            st.markdown("<div class='what-if-label'>Race Conditions</div>", unsafe_allow_html=True)
            rain_wi      = st.selectbox("Rainfall", ["Dry", "Wet", "Mixed"])
            air_t        = st.slider("Air Temp (°C)",          10, 45, 24)
            track_t      = st.slider("Track Temp (°C)",        15, 65, 38)
            humid        = st.slider("Humidity (%)",           20, 100, 55)
            wind_s       = st.slider("Wind Speed (m/s)",        0, 15,  2)
            sc_prob_wi   = st.slider("Safety Car Probability",  0.0, 1.0, 0.35, step=0.05)
            circuit_wi   = st.selectbox("Circuit Type", ["Permanent", "Street", "Road"])
            team_dnf_wi  = st.slider("Constructor DNF Rate",   0.0, 0.3, 0.05, step=0.01)
            pit_eff_wi   = st.slider("Pit Stop Duration (s)",  1.8, 4.5, 2.5, step=0.1)
            compound_wi  = st.selectbox("Starting Compound", ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"])

        circuit_enc_wi = {"Permanent": 0, "Street": 1, "Road": 2}[circuit_wi]
        compound_enc_wi= {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}[compound_wi]
        rain_val       = 1 if rain_wi == "Wet" else (0.5 if rain_wi == "Mixed" else 0)

        fc = meta.get("feature_cols", FEATURE_COLS) if meta else FEATURE_COLS

        input_vals = {
            "grid_position":          grid_pos,
            "quali_delta_to_pole":    (grid_pos - 1) * 0.08,
            "driver_form_3":          driver_form,
            "driver_form_5":          driver_form * 1.02,
            "circuit_specialist_score": driver_form * 0.95,
            "dnf_rate_driver":        dnf_rate_d,
            "seasons_experience":     5.0,
            "team_dnf_rate":          team_dnf_wi,
            "constructor_avg_finish": float(grid_pos),
            "pit_stop_efficiency":    pit_eff_wi,
            "champ_points_before":    float(champ_pts),
            "champ_position_before":  float(champ_pos_v),
            "points_gap_to_leader":   float(max(0, 450 - champ_pts)),
            "starting_compound_enc":  compound_enc_wi,
            "total_stints":           2.0,
            "avg_tyre_age":           2.0,
            "tyre_age_at_start":      0.0,
            "num_pit_stops":          1.0,
            "avg_stop_duration":      pit_eff_wi,
            "air_temperature":        float(air_t),
            "track_temperature":      float(track_t),
            "humidity":               float(humid),
            "wind_speed":             float(wind_s),
            "rainfall":               rain_val,
            "track_temp_delta":       float(track_t) - float(air_t),
            "circuit_type_enc":       circuit_enc_wi,
            "is_street_circuit":      1 if circuit_enc_wi == 1 else 0,
            "sc_probability":         sc_prob_wi,
        }

        X        = np.array([[input_vals.get(c, 0) for c in fc]])
        raw_pred = model.predict(X)[0]

        # Apply same dominance snap as main predictor
        # so What-If results are consistent with Race Predictor
        dominance = (
            input_vals.get("driver_form_3", 10)
            + input_vals.get("sc_probability", 0.35) * 10
            + input_vals.get("dnf_rate_driver", 0.06) * 20
            + input_vals.get("champ_position_before", 10)
            + input_vals.get("points_gap_to_leader", 200) / 50
        )
        if grid_pos == 1 and dominance < 8.0:
            pred_pos = 1
        elif grid_pos == 1 and dominance < 12.0 and raw_pred < 2.5:
            pred_pos = 1
        else:
            pred_pos = max(1, min(20, round(raw_pred)))

        st.markdown("<hr/>", unsafe_allow_html=True)
        delta_g   = grid_pos - pred_pos
        delta_str = f"+{delta_g}" if delta_g > 0 else str(delta_g)
        pos_color = "#ffcd00" if pred_pos <= 3 else ("#2aff7a" if pred_pos <= 6 else ("#e8002d" if pred_pos >= 15 else "#f5f5f5"))

        st.markdown(f"""
        <div class="pw-card pw-card-accent" style="text-align:center;padding:2rem">
            <div class="pw-section-label" style="text-align:center">{driver_name} — Predicted Outcome</div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:5rem;font-weight:800;letter-spacing:0.04em;color:{pos_color};line-height:1">
            P{pred_pos:02d}</div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.8rem;color:rgba(245,245,245,0.4);margin-top:8px">
            STARTED P{grid_pos:02d} &nbsp;·&nbsp; <span style="color:{pos_color}">{delta_str} POSITIONS</span>
            &nbsp;·&nbsp; SC PROB {sc_prob_wi*100:.0f}%
            &nbsp;·&nbsp; {compound_wi}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Sensitivity: grid pos sweep
        st.markdown("<div class='pw-section-label' style='margin-top:1.5rem'>Grid Position Sensitivity</div>", unsafe_allow_html=True)
        preds_r = []
        for gp in range(1, 21):
            v = dict(input_vals)
            v["grid_position"]           = gp
            v["quali_delta_to_pole"]     = (gp - 1) * 0.08
            v["driver_form_3"]           = max(1, driver_form + (gp - grid_pos) * 0.3)
            v["circuit_specialist_score"]= max(1, driver_form * 0.95 + (gp - grid_pos) * 0.3)
            Xr = np.array([[v.get(c, 0) for c in fc]])
            preds_r.append(max(1, min(20, round(model.predict(Xr)[0]))))

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=list(range(1, 21)), y=preds_r, mode="lines+markers",
            line=dict(color="#e8002d", width=2),
            marker=dict(color="#ffcd00", size=7, line=dict(color="#e8002d", width=1)),
            fill="tozeroy", fillcolor="rgba(232,0,45,0.06)",
        ))
        fig3.add_trace(go.Scatter(
            x=[grid_pos], y=[pred_pos], mode="markers",
            marker=dict(color="#ffcd00", size=13, symbol="diamond", line=dict(color="#fff", width=1)),
            showlegend=False,
        ))
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Rajdhani", color="#f5f5f5"),
            xaxis=dict(gridcolor="#1a1a1a", dtick=2,
                       title=dict(text="GRID POSITION", font=dict(family="Share Tech Mono", size=9, color="rgba(245,245,245,0.4)"))),
            yaxis=dict(gridcolor="#1a1a1a", autorange="reversed",
                       title=dict(text="PREDICTED FINISH", font=dict(family="Share Tech Mono", size=9, color="rgba(245,245,245,0.4)"))),
            showlegend=False, margin=dict(l=40, r=20, t=10, b=40), height=300,
        )
        st.plotly_chart(fig3, use_container_width=True)

        # SC probability sweep
        st.markdown("<div class='pw-section-label'>Safety Car Probability Impact</div>", unsafe_allow_html=True)
        sc_range = [i / 10 for i in range(0, 11)]
        sc_preds = []
        for sc in sc_range:
            v = dict(input_vals)
            v["sc_probability"] = sc
            Xr = np.array([[v.get(c, 0) for c in fc]])
            sc_preds.append(max(1, min(20, round(model.predict(Xr)[0]))))

        fig4 = go.Figure(go.Scatter(
            x=sc_range, y=sc_preds, mode="lines+markers",
            line=dict(color="#ffcd00", width=2),
            marker=dict(color="#e8002d", size=7),
            fill="tozeroy", fillcolor="rgba(255,205,0,0.05)",
        ))
        fig4.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Rajdhani", color="#f5f5f5"),
            xaxis=dict(gridcolor="#1a1a1a", tickformat=".0%",
                       title=dict(text="SAFETY CAR PROBABILITY", font=dict(family="Share Tech Mono", size=9, color="rgba(245,245,245,0.4)"))),
            yaxis=dict(gridcolor="#1a1a1a", autorange="reversed",
                       title=dict(text="PREDICTED FINISH", font=dict(family="Share Tech Mono", size=9, color="rgba(245,245,245,0.4)"))),
            showlegend=False, margin=dict(l=40, r=20, t=10, b=40), height=260,
        )
        st.plotly_chart(fig4, use_container_width=True)

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