<div align="center">


# PITWALL INTEL



> **DATA DRIVEN &nbsp;·&nbsp; RACE READY &nbsp;·&nbsp; POWERED BY OPENF1**

<br/>

[![Live App](https://img.shields.io/badge/▶%20LIVE%20APP-pitwall--intel.streamlit.app-E8002D?style=for-the-badge&labelColor=0a0a0a)](https://pitwall-intel.streamlit.app/)
&nbsp;
[![OpenF1](https://img.shields.io/badge/DATA-OPENF1.ORG-FFCD00?style=for-the-badge&labelColor=0a0a0a)](https://openf1.org/)
&nbsp;
[![Python](https://img.shields.io/badge/PYTHON-3.10+-3776AB?style=for-the-badge&labelColor=0a0a0a&logo=python&logoColor=white)](https://python.org/)
&nbsp;
[![XGBoost](https://img.shields.io/badge/MODEL-XGBOOST-00ADD8?style=for-the-badge&labelColor=0a0a0a)](https://xgboost.readthedocs.io/)

<br/>
<br/>

---

<br/>

</div>

## WHAT IS THIS

**PitWall Intel** is an F1 race result predictor that thinks like a race engineer. It pulls 3 seasons of real Formula 1 data from the OpenF1 API, engineers predictive features from qualifying pace, tyre strategy, championship pressure, and weather — then runs an XGBoost model to predict the full race finishing order before lights out.

Built for intermediate data science students who want a real project with real data and results worth showing off.

<br/>

---

<br/>

## SCREENS

<br/>

```
┌─────────────────────────────────────────────────────────────────┐
│  01  RACE PREDICTOR       Select a GP · set grid · run the model│
│  02  MODEL PERFORMANCE    MAE · RMSE · podium accuracy · SHAP   │
│  03  DATA EXPLORER        Browse 3 seasons of race data         │
│  04  WHAT-IF SIMULATOR    Change inputs live · watch it shift   │
└─────────────────────────────────────────────────────────────────┘
```

<br/>

---

<br/>

## SETUP

<br/>

### Step 1 — Create a virtual environment

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

<br/>

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

<br/>

### Step 3 — Launch the app

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

> **Windows shortcut:** double-click `run_windows.bat`
> **Mac/Linux shortcut:** `./run_mac_linux.sh`

<br/>

---

<br/>

## FIRST RUN

<br/>

```
1  Open the app in your browser
   ↓
2  Click "Fetch + Train Model" in the sidebar
   ↓
3  Pipeline runs automatically (3 – 8 min)
   ├─ Downloads 3 seasons from OpenF1 API
   ├─ Engineers all predictive features
   ├─ Trains XGBoost + Random Forest
   └─ Saves models to /models/
   ↓
4  Go to Race Predictor → select a GP → Run Prediction
```

<br/>

> **Note:** 404 and 429 warnings in the terminal are expected — some sessions have no data in OpenF1, and the API rate-limits heavy requests. The pipeline handles both automatically.

<br/>

---

<br/>

## PROJECT STRUCTURE

<br/>

```
pitwall_intel/
│
├── app.py                         Main Streamlit application
├── requirements.txt               Python dependencies
│
├── assets/
│   └── f1_logo.png                F1 logo shown in the header
│
├── utils/
│   ├── data_pipeline.py           OpenF1 fetcher + feature engineering
│   └── model_trainer.py           XGBoost + Random Forest training
│
├── data/
│   ├── raw/                       Raw API responses saved as CSV
│   └── processed/
│       └── features_final.csv     Master feature dataset (~1,300 rows)
│
└── models/
    ├── xgboost_model.pkl          Primary model
    ├── rf_model.pkl               Comparison model
    ├── best_model.pkl             Best performer (auto-selected)
    └── model_meta.pkl             Feature list + evaluation metrics
```

<br/>

---

<br/>

## DATA SOURCES & FEATURES

<br/>

**Source:** [OpenF1 API](https://openf1.org/) — free, no API key, REST-based

**Seasons:** 2023 · 2024 · 2025 &nbsp;|&nbsp; **Coverage:** ~1,300 race-driver rows

<br/>

| Feature | Source | Description |
|---|---|---|
| `grid_position` | Starting Grid | Where the driver starts on the grid |
| `quali_delta_to_pole` | Starting Grid | Gap to pole position in seconds |
| `driver_form_3` | Session Results | Rolling avg of last 3 race finishes |
| `circuit_specialist_score` | Session Results | Driver's historical avg at this circuit |
| `total_stints` | Stints | Number of tyre stints (strategy complexity) |
| `starting_compound` | Stints | Tyre compound at race start |
| `avg_stop_duration` | Pit Stops | Average stationary pit stop time |
| `champ_points_before` | Championship | Points entering the race weekend |
| `champ_position_before` | Championship | Championship standing before the race |
| `team_dnf_rate` | Derived | Constructor reliability score |
| `rainfall` | Weather | Whether rain affected the race |
| `track_temperature` | Weather | Track surface temperature |

<br/>

---

<br/>

## TECH STACK

<br/>

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| UI Framework | Streamlit 1.35.0 |
| Primary Model | XGBoost 2.0.3 |
| Comparison Model | scikit-learn Random Forest |
| Data Processing | pandas 2.2.2 · numpy 1.26.4 |
| Visualisation | Plotly 5.22.0 |
| Explainability | SHAP 0.45.1 |
| Data Source | OpenF1 REST API |
| Styling | Custom CSS — carbon fibre dark theme |

<br/>

---

<br/>

## MODEL NOTES

<br/>

```
Training split   →   Time-based: older seasons train, latest season tests
DNF encoding     →   Did Not Finish treated as position 21
Evaluation       →   MAE (positions off) · RMSE · Podium Accuracy %
Best model       →   Auto-selected between XGBoost and Random Forest
What-If mode     →   Runs inference live on every slider change
```

<br/>

---

<br/>

## AUTHOR

<br/>

<div align="center">

**MUKUL DHATTARWAL**

*Data Science Enthusiast*

<br/>

[![Live App](https://img.shields.io/badge/▶%20LIVE%20APP-pitwall--intel.streamlit.app-E8002D?style=flat-square&labelColor=0a0a0a)](https://pitwall-intel.streamlit.app/)
&nbsp;&nbsp;
[![LinkedIn](https://img.shields.io/badge/LINKEDIN-mukuldhattarwal-0A66C2?style=flat-square&labelColor=0a0a0a&logo=linkedin&logoColor=white)](https://linkedin.com/in/mukuldhattarwal)

</div>

<br/>

---

<div align="center">

<br/>

```
PITWALL INTEL v1.0  ·  FOR EDUCATIONAL USE ONLY  ·  DATA: OPENF1.ORG
```

<br/>

</div>
