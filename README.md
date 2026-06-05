<div align="center">

<h1>🏎️PITWALL INTEL🏁</h1>

<p><strong>Data Driven. Race Ready.</strong></p>

<p>F1 Race Result Predictor with XGBoost, OpenF1 API, and Streamlit</p>

<p>
    <a href="https://pitwall-intel.streamlit.app/"><img src="https://img.shields.io/badge/Live%20App-pitwall--intel.streamlit.app-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live App" /></a>
    <a href="https://openf1.org/"><img src="https://img.shields.io/badge/Data-OpenF1-0F9D58?style=for-the-badge" alt="OpenF1" /></a>
</p>

</div>

## LIVE DEMO

- Streamlit App: [https://pitwall-intel.streamlit.app/](https://pitwall-intel.streamlit.app/)

---

## SETUP

### 1. Create & activate a virtual environment (recommended)

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Windows (cmd):
```powershell
python -m venv .venv
.venv\Scripts\activate.bat
```

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

Locally:
```bash
streamlit run app.py
```

Windows (shortcut):
```powershell
.\run_windows.bat
```

The app will open at the URL printed by Streamlit (typically http://localhost:8501). If a different port is shown, open that URL instead.

---

## FIRST RUN

1. Open the app in your browser
2. Click **"Fetch + Train Model"** in the left sidebar
3. Wait 3–8 minutes while the pipeline:
    - Downloads 3 seasons of F1 data from [OpenF1 API](https://openf1.org/)
   - Engineers features (driver form, quali delta, tyre strategy, etc.)
   - Trains XGBoost and Random Forest models
4. Once complete, go to **Race Predictor** and run predictions

---

## PROJECT STRUCTURE

```
pitwall_intel/
├── app.py                        Main Streamlit application
├── requirements.txt              Python dependencies
├── assets/
│   └── f1_logo.png               F1 logo for header
├── utils/
│   ├── data_pipeline.py          OpenF1 data fetcher + feature engineering
│   └── model_trainer.py          XGBoost + Random Forest training
├── data/
│   ├── raw/                      Raw API response CSVs
│   └── processed/
│       └── features_final.csv    Master feature dataset
└── models/
    ├── xgboost_model.pkl         Trained XGBoost model
    ├── rf_model.pkl              Trained Random Forest model
    ├── best_model.pkl            Best performing model
    └── model_meta.pkl            Feature names + metrics
```

---

## SCREENS

| Screen | Description |
|---|---|
| Race Predictor | Select a GP, set weather + grid, run prediction |
| Model Performance | MAE, RMSE, podium accuracy, feature importance |
| Data Explorer | Browse the full training dataset |
| What-If Simulator | Change inputs live and see prediction shift |

---

## DATA SOURCES

- **[OpenF1 API](https://openf1.org/)** — free, no API key needed
- Sessions, results, grids, stints, pit stops, weather, standings
- Coverage: 2023, 2024, 2025 seasons

---

## TECH STACK

- **Python:** 3.10+
- **Streamlit:** 1.35.0
- **pandas:** 2.2.2
- **numpy:** 1.26.4
- **scikit-learn:** 1.4.2
- **xgboost:** 2.0.3
- **plotly:** 5.22.0
- **shap:** 0.45.1
- See `requirements.txt` for exact dependency list and versions

---

## NOTES

- DNFs are encoded as position 21 in the model
- Time-based train/test split: trains on older seasons, tests on latest
- What-If simulator runs the model live on every input change
- Internet required for data fetch and driver/race info

---

*PitWall Intel v1.0 — For educational use only*

## 👤 Author

<div align="center">

**Mukul Dhattarwal**
Data Science Enthusiast

[![Live App](https://img.shields.io/badge/🚀%20Live%20App-pitwall--intel.streamlit.app-FF4B4B?style=flat-square)](https://pitwall-intel.streamlit.app/)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Mukul-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/mukuldhattarwal)

---
