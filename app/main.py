"""
PitWall Intel — Main Streamlit Application
Race Result Predictor with F1-themed dark UI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import joblib
import json
import os
import time
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PitWall Intel",
    page_icon="app/assets/f1_logo.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_URL = "https://api.openf1.org/v1"

# ── F1 Team Colors ────────────────────────────────────────────────────────────
TEAM_COLORS = {
    "Red Bull Racing":      "#3671C6",
    "Ferrari":              "#E8002D",
    "Mercedes":             "#27F4D2",
    "McLaren":              "#FF8000",
    "Aston Martin":         "#229971",
    "Alpine":               "#FF87BC",
    "Williams":             "#64C4FF",
    "RB":                   "#6692FF",
    "Haas F1 Team":         "#B6BABD",
    "Kick Sauber":          "#52E252",
    "Racing Bulls":         "#6692FF",
    "Sauber":               "#52E252",
}

DEFAULT_COLOR = "#E8002D"

# ── CSS ───────────────────────────────────────────────────────────────────────
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&family=Bebas+Neue&display=swap');

    :root {
        --bg-primary:    #080808;
        --bg-secondary:  #0f0f0f;
        --bg-card:       #111111;
        --bg-card-hover: #161616;
        --accent-red:    #e8002d;
        --accent-yellow: #ffcd00;
        --accent-white:  #f0f0f0;
        --text-primary:  #f0f0f0;
        --text-secondary:#a0a0a0;
        --text-dim:      #555555;
        --border:        #222222;
        --border-accent: #e8002d33;
        --glow-red:      0 0 20px #e8002d44;
        --glow-yellow:   0 0 20px #ffcd0044;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        font-family: 'Rajdhani', sans-serif !important;
    }

    [data-testid="stAppViewContainer"] {
        background-image:
            repeating-linear-gradient(
                0deg,
                transparent,
                transparent 39px,
                #ffffff04 39px,
                #ffffff04 40px
            ),
            repeating-linear-gradient(
                90deg,
                transparent,
                transparent 39px,
                #ffffff04 39px,
                #ffffff04 40px
            );
    }

    [data-testid="stSidebar"] {
        background-color: #0a0a0a !important;
        border-right: 1px solid var(--border) !important;
    }

    [data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
        font-family: 'Rajdhani', sans-serif !important;
    }

    .main-title {
        font-family: 'Bebas Neue', sans-serif !important;
        font-size: 3.4rem !important;
        letter-spacing: 0.12em;
        background: linear-gradient(135deg, #ffffff 0%, #e8002d 60%, #ffcd00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
        display: inline-block;
    }

    .tagline {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.78rem;
        color: var(--text-dim);
        letter-spacing: 0.25em;
        text-transform: uppercase;
        margin-top: 2px;
    }

    .section-header {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.3rem;
        letter-spacing: 0.15em;
        color: var(--accent-red);
        border-bottom: 1px solid var(--border-accent);
        padding-bottom: 6px;
        margin-bottom: 16px;
        text-transform: uppercase;
    }

    .stat-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent-red);
        padding: 16px 20px;
        margin-bottom: 12px;
        position: relative;
        overflow: hidden;
    }

    .stat-card::before {
        content: '';
        position: absolute;
        top: 0; right: 0;
        width: 60px; height: 60px;
        background: radial-gradient(circle at top right, #e8002d11, transparent);
    }

    .stat-label {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.62rem;
        color: var(--text-dim);
        letter-spacing: 0.2em;
        text-transform: uppercase;
    }

    .stat-value {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 2rem;
        color: var(--text-primary);
        line-height: 1.1;
    }

    .stat-value-small {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.3rem;
        color: var(--accent-yellow);
    }

    .driver-row {
        display: flex;
        align-items: center;
        padding: 10px 16px;
        margin-bottom: 4px;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-left: 4px solid var(--team-color, #e8002d);
        transition: all 0.15s ease;
        position: relative;
        overflow: hidden;
    }

    .driver-row:hover {
        background: var(--bg-card-hover);
        border-color: #333333;
    }

    .driver-rank {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.6rem;
        color: var(--text-dim);
        width: 40px;
        text-align: center;
    }

    .driver-rank.top3 { color: var(--accent-yellow); }

    .driver-name {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        font-size: 1.05rem;
        letter-spacing: 0.04em;
        flex: 1;
        margin-left: 12px;
    }

    .driver-team {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.65rem;
        color: var(--text-dim);
        letter-spacing: 0.1em;
    }

    .confidence-bar-wrap {
        width: 120px;
        height: 4px;
        background: #1a1a1a;
        margin-left: 16px;
        position: relative;
        overflow: hidden;
    }

    .confidence-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--accent-red), var(--accent-yellow));
        transition: width 0.4s ease;
    }

    .pred-pos {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-secondary);
        width: 60px;
        text-align: right;
    }

    .alert-upset {
        background: linear-gradient(135deg, #1a0800, #0f0f0f);
        border: 1px solid #ff800033;
        border-left: 3px solid #ff8000;
        padding: 12px 16px;
        margin-bottom: 8px;
    }

    .alert-upset-title {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.65rem;
        color: #ff8000;
        letter-spacing: 0.2em;
        text-transform: uppercase;
    }

    .telemetry-loader {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.72rem;
        color: var(--accent-red);
        letter-spacing: 0.1em;
        animation: blink 1s step-end infinite;
    }

    @keyframes blink {
        50% { opacity: 0; }
    }

    .wet-badge {
        background: #001a33;
        border: 1px solid #0088ff44;
        color: #66aaff;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.6rem;
        padding: 2px 8px;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .dry-badge {
        background: #1a1000;
        border: 1px solid #ffcd0033;
        color: var(--accent-yellow);
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.6rem;
        padding: 2px 8px;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .stSelectbox > div > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        font-family: 'Rajdhani', sans-serif !important;
    }

    .stSlider > div > div > div {
        background: var(--accent-red) !important;
    }

    .stButton > button {
        background: var(--accent-red) !important;
        color: white !important;
        border: none !important;
        font-family: 'Bebas Neue', sans-serif !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.15em !important;
        padding: 10px 28px !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background: #c0001f !important;
        box-shadow: var(--glow-red) !important;
        transform: translateY(-1px) !important;
    }

    div[data-testid="metric-container"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-left: 3px solid var(--accent-red) !important;
        padding: 12px !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-secondary) !important;
        border-bottom: 1px solid var(--border) !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--text-dim) !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        font-size: 0.9rem !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--accent-red) !important;
        border-bottom: 2px solid var(--accent-red) !important;
    }

    .stMarkdown p {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1rem;
    }

    .info-mono {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-dim);
        letter-spacing: 0.08em;
    }

    .compound-soft   { color: #ff3333; font-weight: 700; }
    .compound-medium { color: #ffcd00; font-weight: 700; }
    .compound-hard   { color: #cccccc; font-weight: 700; }
    .compound-inter  { color: #39b54a; font-weight: 700; }
    .compound-wet    { color: #0099ff; font-weight: 700; }

    .title-row {
        display: flex;
        align-items: center;
        gap: 18px;
        margin-bottom: 4px;
    }

    .f1-logo-svg {
        width: 52px;
        height: auto;
        flex-shrink: 0;
        margin-top: -4px;
    }

    /* hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2rem !important; }
    </style>
    """, unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_api(endpoint, params=None):
    try:
        r = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def get_seasons():
    return [2025, 2024, 2023]


@st.cache_data(ttl=3600)
def get_race_meetings(year):
    data = fetch_api("sessions", {"year": year, "session_name": "Race"})
    df = pd.DataFrame(data) if data else pd.DataFrame()
    if df.empty:
        return df
    return df[["meeting_key", "circuit_short_name", "country_name",
               "date_start", "circuit_type", "session_key"]].drop_duplicates("meeting_key")


@st.cache_data(ttl=3600)
def get_drivers_for_session(session_key):
    data = fetch_api("drivers", {"session_key": session_key})
    return pd.DataFrame(data) if data else pd.DataFrame()


@st.cache_data(ttl=3600)
def get_starting_grid(meeting_key):
    sessions = fetch_api("sessions", {"meeting_key": meeting_key, "session_name": "Qualifying"})
    if not sessions:
        return pd.DataFrame()
    qkey = sessions[0]["session_key"]
    data = fetch_api("starting_grid", {"session_key": qkey})
    return pd.DataFrame(data) if data else pd.DataFrame()


@st.cache_data(ttl=3600)
def get_race_result(session_key):
    data = fetch_api("session_result", {"session_key": session_key})
    return pd.DataFrame(data) if data else pd.DataFrame()


@st.cache_data(ttl=3600)
def get_stints_for_session(session_key):
    data = fetch_api("stints", {"session_key": session_key})
    return pd.DataFrame(data) if data else pd.DataFrame()


@st.cache_data(ttl=3600)
def get_weather_for_session(session_key):
    data = fetch_api("weather", {"session_key": session_key})
    if not data:
        return {}
    df = pd.DataFrame(data)
    return {
        "avg_air_temp": df["air_temperature"].mean() if "air_temperature" in df else 25.0,
        "avg_track_temp": df["track_temperature"].mean() if "track_temperature" in df else 35.0,
        "rainfall": int(df["rainfall"].max() > 0) if "rainfall" in df else 0,
        "avg_wind_speed": df["wind_speed"].mean() if "wind_speed" in df else 2.0,
    }


@st.cache_data(ttl=3600)
def get_championship_standings(session_key):
    data = fetch_api("championship_drivers", {"session_key": session_key})
    return pd.DataFrame(data) if data else pd.DataFrame()


def load_model():
    xgb_path = "models/xgb_model.pkl"
    feat_path = "models/feature_cols.json"
    metrics_path = "models/metrics.json"
    if os.path.exists(xgb_path) and os.path.exists(feat_path):
        model = joblib.load(xgb_path)
        with open(feat_path) as f:
            features = json.load(f)
        metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
        return model, features, metrics
    return None, None, {}


# ── Prediction engine ─────────────────────────────────────────────────────────
def build_prediction_input(drivers_df, grid_df, stints_df, weather, champ_df, circuit_type):
    rows = []
    circuit_map = {"Permanent": 0, "Temporary - Street": 1, "Temporary - Road": 2}
    circuit_enc = circuit_map.get(circuit_type, 0)

    pole_time = None
    if not grid_df.empty and "lap_duration" in grid_df.columns:
        pole_time = grid_df["lap_duration"].min()

    for _, driver in drivers_df.iterrows():
        dn = driver["driver_number"]

        # Grid position
        grid_pos = 10
        quali_time = None
        if not grid_df.empty and "driver_number" in grid_df.columns:
            g = grid_df[grid_df["driver_number"] == dn]
            if not g.empty:
                grid_pos = int(g["position"].iloc[0]) if "position" in g.columns else 10
                quali_time = g["lap_duration"].iloc[0] if "lap_duration" in g.columns else None

        quali_delta = (quali_time - pole_time) if (quali_time and pole_time) else 1.5

        # Stints
        n_stops = 1
        compound_enc = 2
        tyre_age = 0
        if not stints_df.empty and "driver_number" in stints_df.columns:
            ds = stints_df[stints_df["driver_number"] == dn]
            if not ds.empty:
                n_stops = max(0, len(ds) - 1)
                first_compound = ds.iloc[0].get("compound", "MEDIUM")
                c_map = {"SOFT": 1, "MEDIUM": 2, "HARD": 3, "INTERMEDIATE": 4, "WET": 5}
                compound_enc = c_map.get(first_compound, 2)
                tyre_age = ds.iloc[0].get("tyre_age_at_start", 0) or 0

        # Championship
        champ_pts = 0
        champ_pos = 10
        if not champ_df.empty and "driver_number" in champ_df.columns:
            cs = champ_df[champ_df["driver_number"] == dn]
            if not cs.empty:
                champ_pts = cs.iloc[0].get("points_start", 0) or 0
                champ_pos = cs.iloc[0].get("position_start", 10) or 10

        rows.append({
            "driver_number": dn,
            "name_acronym": driver.get("name_acronym", str(dn)),
            "full_name": driver.get("full_name", str(dn)),
            "team_name": driver.get("team_name", "Unknown"),
            "team_colour": driver.get("team_colour", "E8002D"),
            "grid_position": grid_pos,
            "quali_delta_to_pole": quali_delta,
            "driver_form_score": grid_pos * 0.8,  # fallback heuristic
            "circuit_type_enc": circuit_enc,
            "rainfall": weather.get("rainfall", 0),
            "avg_track_temp": weather.get("avg_track_temp", 35.0),
            "avg_air_temp": weather.get("avg_air_temp", 25.0),
            "avg_wind_speed": weather.get("avg_wind_speed", 2.0),
            "n_pit_stops": n_stops,
            "starting_compound_enc": compound_enc,
            "tyre_age_at_start": tyre_age,
            "avg_pit_duration": 2.5,
            "championship_points_before": champ_pts,
            "championship_pos_before": champ_pos,
        })
    return pd.DataFrame(rows)


def run_prediction(model, features, input_df):
    available = [c for c in features if c in input_df.columns]
    X = input_df[available].fillna(0)
    preds = model.predict(X)
    df = input_df.copy()
    df["predicted_score"] = preds
    df = df.sort_values("predicted_score").reset_index(drop=True)
    df["predicted_rank"] = range(1, len(df) + 1)
    max_score = df["predicted_score"].max()
    min_score = df["predicted_score"].min()
    rng = max_score - min_score if max_score != min_score else 1
    df["confidence"] = 1 - (df["predicted_score"] - min_score) / rng
    return df


# ── Chart helpers ─────────────────────────────────────────────────────────────
def make_predicted_grid_chart(pred_df):
    fig = go.Figure()
    colors = [f"#{row['team_colour']}" if row['team_colour'] else DEFAULT_COLOR
              for _, row in pred_df.iterrows()]
    fig.add_trace(go.Bar(
        x=pred_df["predicted_rank"],
        y=pred_df["confidence"] * 100,
        marker_color=colors,
        text=pred_df["name_acronym"],
        textposition="outside",
        textfont=dict(family="Rajdhani", size=11, color="#ffffff"),
        hovertemplate="<b>%{text}</b><br>Predicted P%{x}<br>Confidence: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#0f0f0f",
        plot_bgcolor="#0f0f0f",
        font=dict(family="Rajdhani", color="#a0a0a0"),
        xaxis=dict(
            title="Predicted Finishing Position",
            showgrid=False,
            color="#555",
            tickfont=dict(family="Share Tech Mono", size=10),
        ),
        yaxis=dict(
            title="Confidence Score",
            showgrid=True,
            gridcolor="#1a1a1a",
            color="#555",
        ),
        margin=dict(t=30, b=40, l=40, r=20),
        height=320,
        showlegend=False,
    )
    return fig


def make_strategy_chart(stints_df, drivers_df):
    if stints_df.empty:
        return None
    compound_colors = {
        "SOFT": "#e8002d", "MEDIUM": "#ffcd00", "HARD": "#cccccc",
        "INTERMEDIATE": "#39b54a", "WET": "#0099ff",
    }
    merged = stints_df.merge(
        drivers_df[["driver_number", "name_acronym"]],
        on="driver_number", how="left"
    )
    merged["name_acronym"] = merged["name_acronym"].fillna(merged["driver_number"].astype(str))
    fig = go.Figure()
    for compound, grp in merged.groupby("compound"):
        for _, row in grp.iterrows():
            lap_s = row.get("lap_start", 1)
            lap_e = row.get("lap_end", lap_s + 10)
            fig.add_trace(go.Bar(
                x=[lap_e - lap_s],
                y=[row["name_acronym"]],
                base=[lap_s],
                orientation="h",
                marker_color=compound_colors.get(compound, "#555"),
                name=compound,
                showlegend=False,
                hovertemplate=f"<b>{row['name_acronym']}</b><br>{compound}<br>Laps {lap_s}–{lap_e}<extra></extra>",
            ))
    # Legend patches
    for cmp, col in compound_colors.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=col, symbol="square"),
            name=cmp, showlegend=True,
        ))
    fig.update_layout(
        barmode="overlay",
        paper_bgcolor="#0f0f0f",
        plot_bgcolor="#0f0f0f",
        font=dict(family="Rajdhani", color="#a0a0a0"),
        xaxis=dict(title="Lap Number", showgrid=True, gridcolor="#1a1a1a", color="#555"),
        yaxis=dict(showgrid=False, color="#a0a0a0", tickfont=dict(family="Share Tech Mono", size=10)),
        legend=dict(orientation="h", x=0, y=1.08, font=dict(family="Share Tech Mono", size=10)),
        margin=dict(t=40, b=40, l=60, r=20),
        height=400,
    )
    return fig


def make_feature_importance_chart(model, feature_cols):
    if not hasattr(model, "feature_importances_"):
        return None
    importance = model.feature_importances_
    feat_names = feature_cols[:len(importance)]
    df = pd.DataFrame({"feature": feat_names, "importance": importance})
    df = df.sort_values("importance", ascending=True).tail(10)
    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["feature"],
        orientation="h",
        marker=dict(
            color=df["importance"],
            colorscale=[[0, "#1a0a0a"], [0.5, "#e8002d"], [1.0, "#ffcd00"]],
            showscale=False,
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#0f0f0f",
        plot_bgcolor="#0f0f0f",
        font=dict(family="Rajdhani", color="#a0a0a0"),
        xaxis=dict(title="Importance", showgrid=True, gridcolor="#1a1a1a", color="#555"),
        yaxis=dict(showgrid=False, color="#a0a0a0", tickfont=dict(family="Share Tech Mono", size=10)),
        margin=dict(t=20, b=40, l=180, r=20),
        height=300,
    )
    return fig


# ── F1 Logo SVG ───────────────────────────────────────────────────────────────
F1_LOGO_SVG = """
<svg class="f1-logo-svg" viewBox="0 0 120 60" xmlns="http://www.w3.org/2000/svg">
  <!-- F1 wordmark style logo -->
  <rect width="120" height="60" fill="none"/>
  <!-- Red F -->
  <rect x="4" y="8" width="32" height="8" fill="#E8002D"/>
  <rect x="4" y="8" width="8" height="44" fill="#E8002D"/>
  <rect x="4" y="26" width="26" height="8" fill="#E8002D"/>
  <!-- White 1 -->
  <rect x="52" y="8" width="8" height="44" fill="#FFFFFF"/>
  <polygon points="52,8 60,8 60,16 52,16" fill="#FFFFFF"/>
  <rect x="44" y="44" width="24" height="8" fill="#FFFFFF"/>
  <!-- Speed chevron accent -->
  <polygon points="76,8 96,8 120,52 100,52" fill="#E8002D" opacity="0.85"/>
  <polygon points="84,8 100,8 116,44 100,44" fill="#080808"/>
</svg>
"""


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="padding: 12px 0 20px 0; border-bottom: 1px solid #1a1a1a; margin-bottom: 20px;">
            <div class="title-row">
                {F1_LOGO_SVG}
                <div>
                    <div class="main-title" style="font-size:1.8rem">PitWall Intel</div>
                    <div class="tagline" style="font-size:0.6rem">RACE PREDICTION SYSTEM</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Race Control</div>', unsafe_allow_html=True)

        season = st.selectbox("Season", get_seasons(), index=0)
        meetings_df = get_race_meetings(season)

        if meetings_df.empty:
            st.warning("No races found for this season.")
            return None, None, None

        meetings_df = meetings_df.sort_values("date_start")
        race_labels = meetings_df.apply(
            lambda r: f"{r['circuit_short_name']} — {r['country_name']}", axis=1
        ).tolist()
        selected_idx = st.selectbox("Grand Prix", range(len(race_labels)),
                                    format_func=lambda i: race_labels[i])

        selected_race = meetings_df.iloc[selected_idx]

        st.markdown("---")
        st.markdown('<div class="section-header" style="font-size:0.9rem">Wildcard Mode</div>',
                    unsafe_allow_html=True)
        chaos_mode = st.toggle("Safety Car Chaos Mode", value=False)
        if chaos_mode:
            st.markdown(
                '<span class="info-mono" style="color:#ff8000">SC probability injected into simulation</span>',
                unsafe_allow_html=True
            )

        whatif_mode = st.toggle("What-If Simulator", value=False)

        st.markdown("---")
        model, features, metrics = load_model()
        if model:
            st.markdown('<div class="section-header" style="font-size:0.9rem">Model Status</div>',
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div class="info-mono" style="line-height:2">
            STATUS &nbsp;&nbsp;<span style="color:#39b54a">ONLINE</span><br>
            MAE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#ffcd00">{metrics.get('xgb', {}).get('mae', 'N/A')}</span><br>
            RMSE &nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#ffcd00">{metrics.get('xgb', {}).get('rmse', 'N/A')}</span><br>
            TRAIN &nbsp;&nbsp;&nbsp;<span style="color:#a0a0a0">{metrics.get('n_train', 'N/A')} samples</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-mono" style="color:#e8002d">
            MODEL NOT FOUND<br>
            <span style="color:#555; font-size:0.6rem">Run data_pipeline.py then train_model.py</span>
            </div>
            """, unsafe_allow_html=True)

        return selected_race, chaos_mode, whatif_mode


# ── Main header ───────────────────────────────────────────────────────────────
def render_header():
    st.markdown(f"""
    <div style="margin-bottom: 24px;">
        <div class="title-row">
            {F1_LOGO_SVG}
            <div>
                <div class="main-title">PitWall Intel</div>
                <div class="tagline">Data Driven. Race Ready.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Race Control tab ──────────────────────────────────────────────────────────
def render_race_control(race, weather, stints_df, champ_df):
    st.markdown('<div class="section-header">Race Control — Telemetry Feed</div>',
                unsafe_allow_html=True)

    circuit_type = race.get("circuit_type", "Permanent")
    rainfall = weather.get("rainfall", 0)
    track_temp = weather.get("avg_track_temp", 35)
    air_temp = weather.get("avg_air_temp", 25)
    wind = weather.get("avg_wind_speed", 2.0)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        condition_badge = '<span class="wet-badge">WET</span>' if rainfall else '<span class="dry-badge">DRY</span>'
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Condition</div>
            <div style="margin-top:6px">{condition_badge}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Track Temp</div>
            <div class="stat-value">{track_temp:.1f}<span style="font-size:1rem;color:#555">°C</span></div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Air Temp</div>
            <div class="stat-value">{air_temp:.1f}<span style="font-size:1rem;color:#555">°C</span></div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Wind Speed</div>
            <div class="stat-value">{wind:.1f}<span style="font-size:1rem;color:#555"> m/s</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:12px; padding: 12px 16px; background:#0f0f0f; border:1px solid #1a1a1a;">
        <span class="info-mono">
        CIRCUIT &nbsp; {race.get('circuit_short_name', 'N/A').upper()} &nbsp;&nbsp;|&nbsp;&nbsp;
        TYPE &nbsp; {circuit_type.upper()} &nbsp;&nbsp;|&nbsp;&nbsp;
        COUNTRY &nbsp; {race.get('country_name', 'N/A').upper()}
        </span>
    </div>
    """, unsafe_allow_html=True)


# ── Predicted Grid tab ────────────────────────────────────────────────────────
def render_predicted_grid(pred_df, actual_df):
    st.markdown('<div class="section-header">Predicted Finishing Order</div>',
                unsafe_allow_html=True)

    if pred_df is None or pred_df.empty:
        st.info("Run prediction first using the sidebar controls.")
        return

    col_left, col_right = st.columns([3, 2])

    with col_left:
        for _, row in pred_df.iterrows():
            rank = int(row["predicted_rank"])
            team_color = f"#{row['team_colour']}" if row.get('team_colour') else DEFAULT_COLOR
            rank_class = "top3" if rank <= 3 else ""
            conf_pct = row.get("confidence", 0.5) * 100
            conf_width = int(conf_pct)

            actual_pos = ""
            if not actual_df.empty and "driver_number" in actual_df.columns:
                match = actual_df[actual_df["driver_number"] == row["driver_number"]]
                if not match.empty:
                    ap = match.iloc[0].get("position", "")
                    if ap:
                        diff = rank - int(ap)
                        color = "#39b54a" if diff > 0 else "#e8002d" if diff < 0 else "#555"
                        sign = "+" if diff > 0 else ""
                        actual_pos = f'<span style="color:{color};font-family:Share Tech Mono,monospace;font-size:0.65rem">{sign}{diff}</span>'

            st.markdown(f"""
            <div class="driver-row" style="--team-color: {team_color}">
                <div class="driver-rank {rank_class}">{rank}</div>
                <div style="margin-left:12px;flex:1">
                    <div class="driver-name">{row.get('full_name', row.get('name_acronym', ''))}</div>
                    <div class="driver-team">{row.get('team_name', '')}</div>
                </div>
                <div class="confidence-bar-wrap">
                    <div class="confidence-bar-fill" style="width:{conf_width}%"></div>
                </div>
                <div class="pred-pos">{actual_pos}</div>
                <div style="font-family:Share Tech Mono,monospace;font-size:0.65rem;color:#555;width:40px;text-align:right">
                    {conf_pct:.0f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header" style="font-size:0.9rem">Confidence Distribution</div>',
                    unsafe_allow_html=True)
        fig = make_predicted_grid_chart(pred_df)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Strategy tab ──────────────────────────────────────────────────────────────
def render_strategy(stints_df, drivers_df, pred_df):
    st.markdown('<div class="section-header">Tyre Strategy Breakdown</div>',
                unsafe_allow_html=True)

    if stints_df.empty:
        st.info("No stint data available for this session.")
        return

    fig = make_strategy_chart(stints_df, drivers_df)
    if fig:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Compound summary
    st.markdown('<div class="section-header" style="font-size:0.9rem;margin-top:16px">Starting Compound Summary</div>',
                unsafe_allow_html=True)
    compound_map_css = {
        "SOFT": "compound-soft", "MEDIUM": "compound-medium",
        "HARD": "compound-hard", "INTERMEDIATE": "compound-inter", "WET": "compound-wet"
    }
    first_stints = stints_df[stints_df["stint_number"] == 1] if "stint_number" in stints_df.columns else stints_df
    merged = first_stints.merge(drivers_df[["driver_number", "name_acronym", "team_name"]],
                                on="driver_number", how="left")
    cols = st.columns(4)
    for i, (_, row) in enumerate(merged.iterrows()):
        cmp = row.get("compound", "UNKNOWN")
        css_cls = compound_map_css.get(cmp, "")
        with cols[i % 4]:
            st.markdown(f"""
            <div style="padding:8px 12px;background:#0f0f0f;border:1px solid #1a1a1a;margin-bottom:6px;">
                <div class="info-mono" style="font-size:0.6rem">{row.get('name_acronym','')}</div>
                <div class="{css_cls}" style="font-size:0.9rem">{cmp}</div>
            </div>
            """, unsafe_allow_html=True)


# ── Upset Alerts tab ─────────────────────────────────────────────────────────
def render_upset_alerts(pred_df):
    st.markdown('<div class="section-header">Upset Alert — Predicted Surprises</div>',
                unsafe_allow_html=True)

    if pred_df is None or pred_df.empty:
        return

    upsets = pred_df[
        (pred_df["predicted_rank"] <= 5) &
        (pred_df["grid_position"] >= 8)
    ]

    if upsets.empty:
        st.markdown("""
        <div style="padding:16px;background:#0f0f0f;border:1px solid #1a1a1a;">
            <span class="info-mono">No significant upsets predicted for this race.</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        for _, row in upsets.iterrows():
            gain = int(row["grid_position"]) - int(row["predicted_rank"])
            st.markdown(f"""
            <div class="alert-upset">
                <div class="alert-upset-title">Predicted Underdog Alert</div>
                <div style="font-family:Rajdhani,sans-serif;font-size:1.1rem;font-weight:600;margin-top:4px">
                    {row.get('full_name', row.get('name_acronym', ''))}
                    <span style="color:#ff8000;font-size:0.85rem;margin-left:8px">{row.get('team_name','')}</span>
                </div>
                <div class="info-mono" style="margin-top:4px">
                    STARTS P{int(row['grid_position'])} &nbsp; PREDICTED P{int(row['predicted_rank'])} &nbsp;
                    <span style="color:#39b54a">+{gain} POSITIONS GAINED</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Position change table
    st.markdown('<div class="section-header" style="font-size:0.9rem;margin-top:20px">Grid vs Predicted Delta</div>',
                unsafe_allow_html=True)
    if pred_df is not None and not pred_df.empty:
        delta_df = pred_df[["name_acronym", "team_name", "grid_position", "predicted_rank"]].copy()
        delta_df["delta"] = delta_df["grid_position"] - delta_df["predicted_rank"]
        delta_df = delta_df.sort_values("delta", ascending=False)

        fig = go.Figure(go.Bar(
            x=delta_df["name_acronym"],
            y=delta_df["delta"],
            marker_color=delta_df["delta"].apply(
                lambda v: "#39b54a" if v > 0 else "#e8002d" if v < 0 else "#555"
            ),
            hovertemplate="<b>%{x}</b><br>Position Change: %{y}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor="#0f0f0f",
            plot_bgcolor="#0f0f0f",
            font=dict(family="Rajdhani", color="#a0a0a0"),
            xaxis=dict(showgrid=False, color="#555", tickfont=dict(family="Share Tech Mono", size=9)),
            yaxis=dict(showgrid=True, gridcolor="#1a1a1a", color="#555", zeroline=True,
                       zerolinecolor="#333"),
            margin=dict(t=10, b=40, l=40, r=20),
            height=260,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Model Intel tab ───────────────────────────────────────────────────────────
def render_model_intel(model, features, metrics):
    st.markdown('<div class="section-header">Model Intel — Feature Importance</div>',
                unsafe_allow_html=True)

    if model is None:
        st.markdown("""
        <div style="padding:20px;background:#0f0f0f;border:1px solid #1a1a1a;">
            <div class="info-mono" style="color:#e8002d">MODEL NOT TRAINED</div>
            <div class="info-mono" style="color:#555;margin-top:8px">
                Run the pipeline:<br>
                1. python data_pipeline.py<br>
                2. python train_model.py
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    c1, c2, c3 = st.columns(3)
    xgb_m = metrics.get("xgb", {})
    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">XGBoost MAE</div>
            <div class="stat-value-small">{xgb_m.get('mae', 'N/A')}</div>
            <div class="info-mono">positions off on avg</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">XGBoost RMSE</div>
            <div class="stat-value-small">{xgb_m.get('rmse', 'N/A')}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Training Samples</div>
            <div class="stat-value-small">{metrics.get('n_train', 'N/A')}</div>
        </div>""", unsafe_allow_html=True)

    fig = make_feature_importance_chart(model, features)
    if fig:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── What-If Simulator ─────────────────────────────────────────────────────────
def render_whatif(pred_df, model, features, weather):
    st.markdown('<div class="section-header">What-If Simulator</div>',
                unsafe_allow_html=True)

    if pred_df is None or pred_df.empty or model is None:
        st.info("Prediction must be run and model must be loaded first.")
        return

    st.markdown('<div class="info-mono" style="margin-bottom:12px">Adjust race conditions and see how predictions shift.</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        rain_override = st.selectbox("Weather Override", ["No Change", "Force Dry", "Force Wet"])
    with c2:
        sc_laps = st.slider("Safety Car Laps", 0, 10, 0)
    with c3:
        track_temp_override = st.slider("Track Temp Override (°C)", 15, 60, int(weather.get("avg_track_temp", 35)))

    modified_df = pred_df.copy()
    if rain_override == "Force Wet":
        modified_df["rainfall"] = 1
        modified_df["avg_track_temp"] = min(track_temp_override, 25)
    elif rain_override == "Force Dry":
        modified_df["rainfall"] = 0
        modified_df["avg_track_temp"] = track_temp_override
    else:
        modified_df["avg_track_temp"] = track_temp_override

    if sc_laps > 0:
        # SC compresses the field — reduce grid position effect
        modified_df["grid_position"] = modified_df["grid_position"] * (1 - sc_laps * 0.03)

    new_pred = run_prediction(model, features, modified_df)

    st.markdown('<div class="section-header" style="font-size:0.9rem;margin-top:12px">Adjusted Prediction</div>',
                unsafe_allow_html=True)
    cols = st.columns(5)
    for i, row in new_pred.head(5).iterrows():
        with cols[i]:
            team_color = f"#{row['team_colour']}" if row.get('team_colour') else DEFAULT_COLOR
            st.markdown(f"""
            <div style="text-align:center;padding:10px;background:#0f0f0f;border-top:3px solid {team_color}">
                <div class="info-mono" style="color:#555;font-size:0.55rem">P{int(row['predicted_rank'])}</div>
                <div style="font-family:Bebas Neue,sans-serif;font-size:1.3rem">{row.get('name_acronym','')}</div>
                <div class="info-mono" style="font-size:0.55rem">{row.get('team_name','')[:12]}</div>
            </div>
            """, unsafe_allow_html=True)


# ── MAIN APP ──────────────────────────────────────────────────────────────────
def main():
    load_css()
    render_header()

    selected_race, chaos_mode, whatif_mode = render_sidebar()
    if selected_race is None:
        st.warning("No race data available.")
        return

    session_key = int(selected_race["session_key"])
    meeting_key = int(selected_race["meeting_key"])
    circuit_type = selected_race.get("circuit_type", "Permanent")

    # Load data
    with st.spinner(""):
        drivers_df = get_drivers_for_session(session_key)
        grid_df = get_starting_grid(meeting_key)
        stints_df = get_stints_for_session(session_key)
        weather = get_weather_for_session(session_key)
        champ_df = get_championship_standings(session_key)
        actual_df = get_race_result(session_key)

    model, features, metrics = load_model()

    # Build input and predict
    pred_df = None
    if not drivers_df.empty:
        input_df = build_prediction_input(
            drivers_df, grid_df, stints_df, weather, champ_df, circuit_type
        )

        if chaos_mode:
            input_df["grid_position"] = input_df["grid_position"] * np.random.uniform(0.9, 1.1, len(input_df))

        if model is not None:
            pred_df = run_prediction(model, features, input_df)
        else:
            # Fallback: use grid position as prediction
            input_df["predicted_score"] = input_df["grid_position"] + np.random.uniform(-2, 2, len(input_df))
            input_df = input_df.sort_values("predicted_score").reset_index(drop=True)
            input_df["predicted_rank"] = range(1, len(input_df) + 1)
            max_s = input_df["predicted_score"].max()
            min_s = input_df["predicted_score"].min()
            rng = max_s - min_s if max_s != min_s else 1
            input_df["confidence"] = 1 - (input_df["predicted_score"] - min_s) / rng
            pred_df = input_df

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Race Control",
        "Predicted Grid",
        "Strategy",
        "Upset Alerts",
        "Model Intel",
    ])

    with tab1:
        render_race_control(selected_race, weather, stints_df, champ_df)

    with tab2:
        render_predicted_grid(pred_df, actual_df)

    with tab3:
        render_strategy(stints_df, drivers_df, pred_df)

    with tab4:
        render_upset_alerts(pred_df)

    with tab5:
        render_model_intel(model, features, metrics)
        if whatif_mode and pred_df is not None:
            st.markdown("---")
            render_whatif(pred_df, model, features, weather)


if __name__ == "__main__":
    main()
