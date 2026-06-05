"""
PitWall Intel — Model Training
Trains XGBoost and Random Forest models on F1 race data.
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "grid_position",
    "quali_delta_to_pole",
    "driver_form_score",
    "circuit_type_enc",
    "rainfall",
    "avg_track_temp",
    "avg_air_temp",
    "avg_wind_speed",
    "n_pit_stops",
    "starting_compound_enc",
    "tyre_age_at_start",
    "avg_pit_duration",
    "championship_points_before",
    "championship_pos_before",
]

TARGET_COL = "finish_position_adj"


def load_and_prepare(path="data/processed/master_features.csv"):
    df = pd.read_csv(path)

    # Ensure target
    if "finish_position_adj" not in df.columns:
        df["finish_position_adj"] = df.apply(
            lambda r: 21 if any([
                str(r.get("dnf", False)).lower() == "true",
                str(r.get("dns", False)).lower() == "true",
                str(r.get("dsq", False)).lower() == "true"
            ]) else pd.to_numeric(r.get("position", 20), errors="coerce"),
            axis=1
        )

    df["finish_position_adj"] = pd.to_numeric(df["finish_position_adj"], errors="coerce")
    df = df.dropna(subset=["finish_position_adj"])

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    print(f"Available features: {available_features}")

    X = df[available_features].copy()
    y = df[TARGET_COL].copy()

    # Fill missing values
    X = X.fillna(X.median(numeric_only=True))

    return X, y, df, available_features


def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

    print("\nTraining Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

    # Podium accuracy (top 3)
    def podium_accuracy(y_true, y_pred):
        true_podium = set(np.argsort(y_true.values)[:3])
        pred_podium = set(np.argsort(y_pred)[:3])
        return len(true_podium & pred_podium) / 3

    # Evaluate on multiple test groups
    unique_meetings = []
    xgb_podium_scores = []
    if "meeting_key" in X.columns:
        pass  # handled below

    metrics = {
        "xgb": {"mae": round(xgb_mae, 3), "rmse": round(xgb_rmse, 3)},
        "rf": {"mae": round(rf_mae, 3), "rmse": round(rf_rmse, 3)},
        "features": list(X.columns),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    print(f"\n--- Model Evaluation ---")
    print(f"XGBoost  | MAE: {xgb_mae:.3f} | RMSE: {xgb_rmse:.3f}")
    print(f"RandomForest | MAE: {rf_mae:.3f} | RMSE: {rf_rmse:.3f}")

    return xgb_model, rf_model, metrics, X.columns.tolist()


def save_models(xgb_model, rf_model, metrics, feature_cols, out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(xgb_model, f"{out_dir}/xgb_model.pkl")
    joblib.dump(rf_model, f"{out_dir}/rf_model.pkl")
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{out_dir}/feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    print(f"\nModels saved to {out_dir}/")


def predict_race(input_df, model_path="models/xgb_model.pkl", feature_path="models/feature_cols.json"):
    """Predict finishing positions for a given race input dataframe."""
    model = joblib.load(model_path)
    with open(feature_path) as f:
        feature_cols = json.load(f)
    available = [c for c in feature_cols if c in input_df.columns]
    X = input_df[available].fillna(input_df[available].median(numeric_only=True))
    preds = model.predict(X)
    input_df = input_df.copy()
    input_df["predicted_position"] = preds
    input_df = input_df.sort_values("predicted_position").reset_index(drop=True)
    input_df["predicted_rank"] = range(1, len(input_df) + 1)
    return input_df


if __name__ == "__main__":
    print("=== PitWall Intel — Model Training ===\n")
    X, y, df, features = load_and_prepare()
    xgb_model, rf_model, metrics, fcols = train_models(X, y)
    save_models(xgb_model, rf_model, metrics, fcols)
    print("\nTraining complete.")
