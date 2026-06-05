"""
PitWall Intel — Model Training
Trains XGBoost + Random Forest on F1 historical data.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")

FEATURE_COLS = [
    "grid_position",
    "quali_delta_to_pole",
    "total_stints",
    "avg_tyre_age",
    "num_pit_stops",
    "avg_stop_duration",
    "air_temperature",
    "track_temperature",
    "humidity",
    "wind_speed",
    "rainfall",
    "champ_points_before",
    "champ_position_before",
    "team_dnf_rate",
    "driver_form_3",
    "circuit_specialist_score",
    "circuit_type_enc",
    "compound_enc",
]

TARGET_COL = "finish_position"


def load_features():
    path = os.path.join(PROCESSED_DIR, "features_final.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Feature file not found at {path}. Run data_pipeline.py first."
        )
    df = pd.read_csv(path)
    return df


def prepare_data(df):
    """Clean and prepare features + target for training."""
    df = df.copy()

    # Drop rows with no grid position or finish position
    df = df.dropna(subset=["grid_position", TARGET_COL])

    # Fill remaining numeric NaNs with median
    for col in FEATURE_COLS:
        if col in df.columns:
            if df[col].dtype in [float, int, "float64", "int64"]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(0)
        else:
            df[col] = 0

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    # Time-based split: train on earlier races, test on most recent season
    if "year" in df.columns:
        max_year = df["year"].max()
        train_mask = df["year"] < max_year
        test_mask = df["year"] == max_year

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        if len(X_test) == 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    return X_train, X_test, y_train, y_test, df


def train_xgboost(X_train, y_train):
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, name="Model"):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    # Podium accuracy: predicted top-3 vs actual top-3
    actual_top3 = set(y_test[y_test <= 3].index.tolist())
    pred_series = pd.Series(preds, index=y_test.index)
    pred_top3 = set(pred_series.nsmallest(3).index.tolist())
    podium_acc = len(actual_top3 & pred_top3) / max(len(actual_top3), 1)

    print(f"\n  {name}")
    print(f"    MAE              : {mae:.2f} positions")
    print(f"    RMSE             : {rmse:.2f} positions")
    print(f"    Podium Accuracy  : {podium_acc * 100:.1f}%")
    return {"mae": mae, "rmse": rmse, "podium_acc": podium_acc}


def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_names)
        return imp.sort_values(ascending=False)
    return pd.Series()


def train_and_save():
    print("=" * 55)
    print("  PITWALL INTEL — Model Training")
    print("=" * 55)

    print("\n[1] Loading feature data...")
    df = load_features()
    print(f"    {len(df)} rows loaded")

    print("[2] Preparing train/test split (time-based)...")
    X_train, X_test, y_train, y_test, df_clean = prepare_data(df)
    print(f"    Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    print("[3] Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)

    print("[4] Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)

    print("\n[5] Evaluation Results:")
    print("-" * 40)
    xgb_metrics = evaluate(xgb_model, X_test, y_test, "XGBoost")
    rf_metrics = evaluate(rf_model, X_test, y_test, "Random Forest")

    # Pick best model
    best_model = xgb_model if xgb_metrics["mae"] <= rf_metrics["mae"] else rf_model
    best_name = "XGBoost" if xgb_metrics["mae"] <= rf_metrics["mae"] else "Random Forest"
    print(f"\n  Best model: {best_name}")

    print("\n[6] Feature Importance (XGBoost):")
    imp = get_feature_importance(xgb_model, FEATURE_COLS)
    for feat, score in imp.head(8).items():
        bar = "=" * int(score * 50)
        print(f"    {feat:<30} {bar} {score:.4f}")

    print("\n[7] Saving models...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgboost_model.pkl"))
    joblib.dump(rf_model, os.path.join(MODEL_DIR, "rf_model.pkl"))
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))

    # Save feature names and metrics
    meta = {
        "feature_cols": FEATURE_COLS,
        "best_model": best_name,
        "xgb_mae": xgb_metrics["mae"],
        "rf_mae": rf_metrics["mae"],
        "xgb_podium_acc": xgb_metrics["podium_acc"],
    }
    joblib.dump(meta, os.path.join(MODEL_DIR, "model_meta.pkl"))

    print(f"    Models saved to: {MODEL_DIR}")
    print("\n  Training complete.")
    print("=" * 55)

    return best_model, FEATURE_COLS


if __name__ == "__main__":
    train_and_save()
