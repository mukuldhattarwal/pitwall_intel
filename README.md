<div align="center">

<br/>

<img src="assets/f1_logo.png" alt="PitWall Intel" width="80"/>

# PITWALL INTEL

### F1 Race Result Predictor

**DATA DRIVEN &nbsp;·&nbsp; RACE READY &nbsp;·&nbsp; POWERED BY OPENF1 + FASTF1**

<br/>

[![Live App](https://img.shields.io/badge/▶%20LIVE%20APP-pitwall--intel.streamlit.app-E8002D?style=for-the-badge&labelColor=0a0a0a)](https://pitwall-intel.streamlit.app/)
&nbsp;
[![OpenF1](https://img.shields.io/badge/DATA-OPENF1.ORG-FFCD00?style=for-the-badge&labelColor=0a0a0a)](https://openf1.org/)
&nbsp;
[![FastF1](https://img.shields.io/badge/TELEMETRY-FASTF1-2AFF7A?style=for-the-badge&labelColor=0a0a0a)](https://docs.fastf1.dev/)
&nbsp;
[![Python](https://img.shields.io/badge/PYTHON-3.10+-3776AB?style=for-the-badge&labelColor=0a0a0a&logo=python&logoColor=white)](https://python.org/)
&nbsp;
[![XGBoost](https://img.shields.io/badge/MODEL-XGBOOST-00ADD8?style=for-the-badge&labelColor=0a0a0a)](https://xgboost.readthedocs.io/)

<br/>

---

</div>

## Overview

**PitWall Intel** is an F1 race result predictor that thinks like a race engineer. It pulls 4 seasons of real Formula 1 data from the **OpenF1 API**, enriches it with deep lap-by-lap telemetry from **FastF1** (lap consistency, tyre degradation, sector deltas, throttle telemetry, qualifying gaps), engineers 37+ predictive features from qualifying pace, tyre strategy, championship pressure, pit efficiency, and weather then runs an **XGBoost model** to predict the full 20-car race finishing order before lights out.

Built for data science enthusiasts who want a real, end-to-end ML project with real Formula 1 data and results worth showing off.

<br/>

---

## Screens

```
┌─────────────────────────────────────────────────────────────────────┐
│  01  RACE PREDICTOR       Select a GP · customise grid · run model  │
│  02  MODEL PERFORMANCE    MAE · podium accuracy · feature importance│
│  03  DATA EXPLORER        Browse 4 seasons of race + telemetry data │
│  04  WHAT-IF SIMULATOR    Adjust any variable live · watch it shift │
└─────────────────────────────────────────────────────────────────────┘
```

**Race Predictor** — Choose any 2026 Grand Prix, set starting grid positions, adjust weather conditions, and fire a full prediction. The app fetches live championship standings, tyre stint data, and pit stop efficiency from the OpenF1 API in real time before inference.

**Model Performance** — Inspect MAE, podium accuracy, and a full feature importance chart colour-coded by feature type (standard vs. FastF1 telemetry). Includes a feature coverage table showing every input's live vs. prior vs. default source.

**Data Explorer** — Browse and filter all training data across seasons and circuits. A dedicated telemetry tab shows FastF1-enriched columns with summary statistics. Includes a grid-position vs. finish-position scatter plot coloured by team.

**What-If Simulator** — Drag any slider (grid position, championship points, SC probability, tyre compound, qualifying gap, tyre degradation rate, and more) and the model re-runs inference instantly. Grid sensitivity and SC probability sweep charts update live.

<br/>

---

## Setup

### Step 1 — Clone the repository

```bash
git clone https://github.com/mukuldhattarwal/pitwall-intel.git
cd pitwall-intel
```

### Step 2 — Create a virtual environment

<details>
<summary><strong>Windows (PowerShell)</strong></summary>

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

</details>

<details>
<summary><strong>Windows (cmd)</strong></summary>

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

</details>

<details>
<summary><strong>macOS / Linux</strong></summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
```

</details>

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Launch the app

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

> **Windows shortcut:** double-click `run_windows.bat`  
> **Mac/Linux shortcut:** `./run_mac_linux.sh`

<br/>

---

## First Run

```
1  Open the app in your browser
       ↓
2  Click "Fetch + Train Model" in the sidebar
       ↓
3  The OpenF1 pipeline runs automatically  (~3 – 8 min)
   ├─ Downloads sessions, results & grids (2023–2025)
   ├─ Fetches tyre stints, pit stops & weather
   ├─ Loads championship standings
   ├─ Engineers all 37 predictive features
   ├─ Trains XGBoost + Random Forest
   └─ Saves models to /models/
       ↓
4  (Optional) Click "Fetch Telemetry (FastF1)"
   └─ Adds 9 deep telemetry features: quali_gap_ms · sector deltas ·
      lap_consistency · race_pace_delta · tyre_deg_rate · throttle_pct · sc_laps_actual
       ↓
5  (If telemetry fetched) Click "Fetch + Train Model" again
   └─ Retrains the model with the enriched 37-feature set
       ↓
6  Go to Race Predictor → select a GP → Run Prediction
```

> **Note:** 404 and 429 warnings in the terminal are expected — some sessions have no data in OpenF1, and the API rate-limits heavy requests. The pipeline handles both automatically with built-in fallbacks.

<br/>

---

## Project Structure

```
pitwall_intel/
│
├── app.py                         Main Streamlit application (all 4 pages)
├── requirements.txt               Python dependencies
│
├── assets/
│   └── f1_logo.png                F1 logo rendered in the app header
│
├── utils/
│   ├── data_pipeline.py           OpenF1 fetcher + feature engineering
│   ├── model_trainer.py           XGBoost + Random Forest training + evaluation
│   └── fastf1_pipeline.py         FastF1 telemetry fetcher + feature merger
│
├── data/
│   ├── raw/                       Raw API responses saved as CSV
│   └── processed/
│       └── features_final.csv     Master feature dataset (~1,500 rows)
│
└── models/
    ├── xgboost_model.pkl          Primary prediction model
    ├── rf_model.pkl               Comparison model
    ├── best_model.pkl             Best performer (auto-selected at training time)
    └── model_meta.pkl             Feature list + evaluation metrics
```

<br/>

---

## Data Sources & Features

**Primary source:** [OpenF1 API](https://openf1.org/) — free, no API key required, REST-based  
**Telemetry source:** [FastF1](https://docs.fastf1.dev/) — lap-level and car-level telemetry  
**Seasons covered:** 2023 · 2024 · 2025 · 2026 &nbsp;|&nbsp; **~1,500+ race-driver rows**

### Standard Features (OpenF1 + Derived)

| Feature | Source | Description |
|---|---|---|
| `grid_position` | Starting Grid | Where the driver starts on race day |
| `quali_delta_to_pole` | Derived | Estimated gap to pole in seconds |
| `driver_form_3` / `driver_form_5` | Session Results | Rolling avg finish over last 3 and 5 races |
| `circuit_specialist_score` | Session Results | Driver's historical avg finish at this circuit |
| `total_stints` | OpenF1 Stints | Number of tyre stints (strategy indicator) |
| `starting_compound_enc` | OpenF1 Stints | Encoded tyre compound at race start |
| `avg_tyre_age` / `tyre_age_at_start` | OpenF1 Stints | Tyre age profile across the race |
| `num_pit_stops` / `avg_stop_duration` | OpenF1 Pit API | Stop count and stationary time efficiency |
| `champ_points_before` | OpenF1 Championship | Points entering the race weekend |
| `champ_position_before` | OpenF1 Championship | Championship standing entering the weekend |
| `points_gap_to_leader` | Derived | Point deficit to current championship leader |
| `constructor_avg_finish` | Derived | Team's rolling average finishing position |
| `team_dnf_rate` / `dnf_rate_driver` | Derived | Constructor and driver reliability scores |
| `pit_stop_efficiency` | OpenF1 Pit API | Team's average stationary time |
| `seasons_experience` | Derived | Number of seasons the driver has competed |
| `sc_probability` | Race Control + Historical | Safety car likelihood for this circuit |
| `circuit_type_enc` / `is_street_circuit` | OpenF1 Meetings | Permanent / street / road circuit encoding |
| `air_temperature` / `track_temperature` | User Input + Weather | Race-day thermal conditions |
| `rainfall` / `humidity` / `wind_speed` | User Input | Weather modifiers |
| `track_temp_delta` | Derived | Track minus air temperature gap |

### FastF1 Telemetry Features

| Feature | Source | Description |
|---|---|---|
| `quali_gap_ms` | FastF1 Qualifying | Gap to pole position in milliseconds |
| `s1_delta` / `s2_delta` / `s3_delta` | FastF1 Qualifying | Per-sector time delta to pole |
| `lap_consistency` | FastF1 Race Laps | Standard deviation of lap times (lower = more consistent) |
| `race_pace_delta` | FastF1 Race Laps | Median race pace gap to the fastest driver |
| `tyre_deg_rate` | FastF1 Race Laps | Lap time increase per lap on a stint (ms/lap) |
| `throttle_pct` | FastF1 Car Telemetry | Percentage of lap on full throttle |
| `sc_laps_actual` | FastF1 Track Status | Actual safety car laps in the race |

<br/>

---

## Live Data Flow

On every prediction, the app makes five live API calls before running inference:

```
OpenF1 /v1/championship_drivers    → g champ_points_before · position · gap_to_leader
OpenF1 /v1/stints                  →  starting_compound · total_stints · avg_tyre_age
OpenF1 /v1/pit                     →  num_pit_stops · avg_stop_duration
OpenF1 /v1/race_control            →  sc_probability (live circuit history)
OpenF1 /v1/drivers                 →  current driver lineup + team assignments
```

Where live data is unavailable, the app falls back to the most recent value from the training dataset, then to a sensible default — in that priority order.

<br/>

---

## Model Notes

```
Training split    →  Time-based: 2023–2024 train, 2025 test (no data leakage)
DNF encoding      →  Did Not Finish treated as finishing position 21
Target variable   →  Predicted race finishing position (regression)
Evaluation        →  MAE (positions off) · Podium Accuracy % (P1–P3 hit rate)
Model selection   →  Auto-selects best between XGBoost and Random Forest at training time
Telemetry impact  →  9 FastF1 features added when telemetry pipeline is run
What-If mode      →  Runs full model inference on every slider change, no caching
Pole logic        →  Post-prediction dominance adjustment for pole sitter when form is strong
```

<br/>

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.10+ |
| UI Framework | Streamlit | 1.35.0 |
| Primary Model | XGBoost | 2.0.3 |
| Comparison Model | scikit-learn Random Forest | 1.4.2 |
| Data Processing | pandas · numpy | 2.2.2 · 1.26.4 |
| Visualisation | Plotly | 5.22.0 |
| Telemetry | FastF1 | 3.3+ |
| Data Source | OpenF1 REST API | — |
| Model Serialisation | joblib | — |
| Styling | Custom CSS (carbon-dark theme) | — |

<br/>

---

## Known Limitations

- Predictions are based on historical patterns and statistical priors — they are not a simulation of race physics or strategy.
- FastF1 data is only available for completed sessions; 2026 race telemetry accumulates throughout the season.
- OpenF1 may return 404 for sessions with missing data or 429 when rate-limited. Both are handled gracefully.
- The 2026 driver lineup (Cadillac, Audi entries) uses fallback data until OpenF1 populates live records.
- This project is for educational purposes only and is not affiliated with Formula 1, the FIA, or any constructor.

<br/>

---

## Author

<div align="center">

**MUKUL DHATTARWAL**

*Data Science Enthusiast*

<br/>

[![Live App](https://img.shields.io/badge/▶%20LIVE%20APP-pitwall--intel.streamlit.app-E8002D?style=flat-square&labelColor=0a0a0a)](https://pitwall-intel.streamlit.app/)
&nbsp;&nbsp;
[![LinkedIn](https://img.shields.io/badge/LINKEDIN-mukuldhattarwal-0A66C2?style=flat-square&labelColor=0a0a0a&logo=linkedin&logoColor=white)](https://linkedin.com/in/mukuldhattarwal)

<br/>

---

```
PITWALL INTEL v1.1  ·  FOR EDUCATIONAL USE ONLY  ·  DATA: OPENF1.ORG + FASTF1
```

</div>