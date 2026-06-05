"""
PitWall Intel — Data Pipeline
Fetches and processes F1 data from OpenF1 API for model training.
"""

import requests
import pandas as pd
import numpy as np
import json
import time
import os
from datetime import datetime

BASE_URL = "https://api.openf1.org/v1"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def fetch(endpoint, params=None, retries=3):
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [WARN] Failed {endpoint} params={params}: {e}")
                return []
            time.sleep(2)
    return []


def get_race_sessions(years):
    """Get all race session keys for given years."""
    all_sessions = []
    for year in years:
        data = fetch("sessions", {"year": year, "session_type": "Race", "session_name": "Race"})
        for s in data:
            all_sessions.append({
                "year": year,
                "meeting_key": s.get("meeting_key"),
                "session_key": s.get("session_key"),
                "circuit_short_name": s.get("circuit_short_name"),
                "circuit_key": s.get("circuit_key"),
                "country_name": s.get("country_name"),
                "location": s.get("location"),
                "date_start": s.get("date_start"),
            })
    return pd.DataFrame(all_sessions)


def get_qualifying_sessions(years):
    """Get qualifying session keys."""
    all_q = []
    for year in years:
        data = fetch("sessions", {"year": year, "session_name": "Qualifying"})
        for s in data:
            all_q.append({
                "year": year,
                "meeting_key": s.get("meeting_key"),
                "session_key": s.get("session_key"),
            })
    return pd.DataFrame(all_q)


def get_session_results(session_keys):
    """Fetch race results for each session."""
    results = []
    for sk in session_keys:
        data = fetch("session_result", {"session_key": sk})
        for r in data:
            r["session_key"] = sk
            results.append(r)
        time.sleep(0.3)
    return pd.DataFrame(results)


def get_starting_grids(meeting_keys_sessions):
    """Fetch starting grids (from qualifying sessions of same meeting)."""
    grids = []
    for meeting_key, q_session_key in meeting_keys_sessions:
        data = fetch("starting_grid", {"session_key": q_session_key})
        for g in data:
            g["meeting_key"] = meeting_key
            grids.append(g)
        time.sleep(0.3)
    return pd.DataFrame(grids)


def get_stints(session_keys):
    """Fetch tyre stint data."""
    stints = []
    for sk in session_keys:
        data = fetch("stints", {"session_key": sk})
        for s in data:
            s["session_key"] = sk
            stints.append(s)
        time.sleep(0.3)
    return pd.DataFrame(stints)


def get_pit_stops(session_keys):
    """Fetch pit stop data."""
    pits = []
    for sk in session_keys:
        data = fetch("pit", {"session_key": sk})
        for p in data:
            p["session_key"] = sk
            pits.append(p)
        time.sleep(0.3)
    return pd.DataFrame(pits)


def get_weather(session_keys):
    """Fetch weather per session — take median values."""
    weather_rows = []
    for sk in session_keys:
        data = fetch("weather", {"session_key": sk})
        if data:
            df = pd.DataFrame(data)
            row = {
                "session_key": sk,
                "air_temperature": df["air_temperature"].median() if "air_temperature" in df else np.nan,
                "track_temperature": df["track_temperature"].median() if "track_temperature" in df else np.nan,
                "humidity": df["humidity"].median() if "humidity" in df else np.nan,
                "wind_speed": df["wind_speed"].median() if "wind_speed" in df else np.nan,
                "rainfall": int(df["rainfall"].sum() > 0) if "rainfall" in df else 0,
            }
            weather_rows.append(row)
        time.sleep(0.3)
    return pd.DataFrame(weather_rows)


def get_championship_standings(session_keys):
    """Fetch driver championship standings per race session."""
    standings = []
    for sk in session_keys:
        data = fetch("championship_drivers", {"session_key": sk})
        for d in data:
            d["session_key"] = sk
            standings.append(d)
        time.sleep(0.3)
    return pd.DataFrame(standings)


def get_drivers(session_keys):
    """Fetch driver info from sessions."""
    drivers = []
    seen = set()
    for sk in session_keys:
        data = fetch("drivers", {"session_key": sk})
        for d in data:
            key = (d.get("driver_number"), d.get("session_key"))
            if key not in seen:
                seen.add(key)
                drivers.append({
                    "driver_number": d.get("driver_number"),
                    "name_acronym": d.get("name_acronym"),
                    "full_name": d.get("full_name"),
                    "team_name": d.get("team_name"),
                    "team_colour": d.get("team_colour"),
                    "session_key": sk,
                })
        time.sleep(0.2)
    return pd.DataFrame(drivers)


def get_meetings(years):
    """Fetch meeting/circuit info."""
    meetings = []
    for year in years:
        data = fetch("meetings", {"year": year})
        for m in data:
            meetings.append({
                "meeting_key": m.get("meeting_key"),
                "circuit_short_name": m.get("circuit_short_name"),
                "circuit_type": m.get("circuit_type", "Permanent"),
                "country_name": m.get("country_name"),
                "year": year,
            })
    return pd.DataFrame(meetings)


def safe_col(df, col, default=np.nan):
    """Return column if it exists, else a Series of default."""
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def engineer_features(race_sessions, results, grids, stints, pits, weather, standings, drivers, meetings):
    """
    Build one row per (race, driver) with all engineered features.
    Returns master feature DataFrame ready for ML.
    """

    # Ensure meeting_key exists in results — join from race_sessions via session_key
    results = results.copy()
    if "meeting_key" not in results.columns:
        sk_to_mk = race_sessions.set_index("session_key")["meeting_key"].to_dict()
        results["meeting_key"] = results["session_key"].map(sk_to_mk)

    # Merge results with session metadata
    session_meta_cols = ["session_key", "meeting_key", "circuit_short_name",
                         "country_name", "year", "date_start"]
    session_meta_cols = [c for c in session_meta_cols if c in race_sessions.columns]
    df = results.merge(race_sessions[session_meta_cols], on="session_key", how="left",
                       suffixes=("", "_sess"))

    # If meeting_key came from both sides, consolidate
    if "meeting_key_sess" in df.columns:
        df["meeting_key"] = df["meeting_key"].combine_first(df["meeting_key_sess"])
        df.drop(columns=["meeting_key_sess"], inplace=True)

    # Ensure core columns exist
    for col in ["meeting_key", "circuit_short_name", "country_name", "year", "date_start"]:
        if col not in df.columns:
            df[col] = np.nan

    # --- Driver info merge ---
    if not drivers.empty and "driver_number" in drivers.columns:
        driver_map = (
            drivers.sort_values("session_key")
            .drop_duplicates(subset=["driver_number"], keep="last")
        )
        drv_cols = [c for c in ["driver_number", "name_acronym", "full_name", "team_name"]
                    if c in driver_map.columns]
        df = df.merge(driver_map[drv_cols], on="driver_number", how="left")
    else:
        df["name_acronym"] = df.get("driver_number", "").astype(str)
        df["full_name"] = df.get("driver_number", "").astype(str)
        df["team_name"] = "Unknown"

    # --- Circuit type from meetings ---
    if not meetings.empty and "meeting_key" in meetings.columns and "circuit_type" in meetings.columns:
        meet_clean = meetings[["meeting_key", "circuit_type"]].drop_duplicates("meeting_key")
        df = df.merge(meet_clean, on="meeting_key", how="left")
    else:
        df["circuit_type"] = "Permanent"

    df["circuit_type"] = df["circuit_type"].fillna("Permanent")

    # --- Grid position from starting grid ---
    if not grids.empty and "position" in grids.columns and "meeting_key" in grids.columns:
        grid_clean = grids.rename(columns={
            "position": "grid_position",
            "lap_duration": "qualifying_time"
        }).copy()
        gc_cols = [c for c in ["meeting_key", "driver_number", "grid_position", "qualifying_time"]
                   if c in grid_clean.columns]
        grid_clean = grid_clean[gc_cols].drop_duplicates(["meeting_key", "driver_number"])
        df = df.merge(grid_clean, on=["meeting_key", "driver_number"], how="left")
    else:
        df["grid_position"] = np.nan
        df["qualifying_time"] = np.nan

    if "qualifying_time" not in df.columns:
        df["qualifying_time"] = np.nan
    if "grid_position" not in df.columns:
        df["grid_position"] = np.nan

    # Qualifying delta to pole
    if "meeting_key" in df.columns and df["meeting_key"].notna().any():
        pole_times = (
            df.groupby("meeting_key")["qualifying_time"]
            .min()
            .reset_index()
            .rename(columns={"qualifying_time": "pole_time"})
        )
        df = df.merge(pole_times, on="meeting_key", how="left")
        df["quali_delta_to_pole"] = df["qualifying_time"] - df["pole_time"]
    else:
        df["quali_delta_to_pole"] = np.nan

    # --- Tyre strategy features ---
    if not stints.empty and "session_key" in stints.columns and "driver_number" in stints.columns:
        agg_dict = {}
        if "stint_number" in stints.columns:
            agg_dict["total_stints"] = ("stint_number", "max")
        if "compound" in stints.columns:
            agg_dict["starting_compound"] = ("compound", "first")
        if "tyre_age_at_start" in stints.columns:
            agg_dict["avg_tyre_age"] = ("tyre_age_at_start", "mean")

        if agg_dict:
            stint_agg = stints.groupby(["session_key", "driver_number"]).agg(**agg_dict).reset_index()
            df = df.merge(stint_agg, on=["session_key", "driver_number"], how="left")

    for col, default in [("total_stints", np.nan), ("starting_compound", "MEDIUM"), ("avg_tyre_age", np.nan)]:
        if col not in df.columns:
            df[col] = default

    # --- Pit stop features ---
    if not pits.empty and "session_key" in pits.columns and "driver_number" in pits.columns:
        agg_dict = {}
        if "lap_number" in pits.columns:
            agg_dict["num_pit_stops"] = ("lap_number", "count")
        if "stop_duration" in pits.columns:
            agg_dict["avg_stop_duration"] = ("stop_duration", "mean")
        elif "lane_duration" in pits.columns:
            agg_dict["avg_stop_duration"] = ("lane_duration", "mean")

        if agg_dict:
            pit_agg = pits.groupby(["session_key", "driver_number"]).agg(**agg_dict).reset_index()
            df = df.merge(pit_agg, on=["session_key", "driver_number"], how="left")

    for col in ["num_pit_stops", "avg_stop_duration"]:
        if col not in df.columns:
            df[col] = np.nan

    # --- Weather ---
    if not weather.empty and "session_key" in weather.columns:
        df = df.merge(weather, on="session_key", how="left")

    for col, default in [("air_temperature", 25.0), ("track_temperature", 35.0),
                          ("humidity", 55.0), ("wind_speed", 2.0), ("rainfall", 0)]:
        if col not in df.columns:
            df[col] = default

    # --- Championship standings ---
    if not standings.empty and "session_key" in standings.columns:
        stand_cols_needed = ["session_key", "driver_number"]
        pts_col = next((c for c in ["points_start", "points_current"] if c in standings.columns), None)
        pos_col = next((c for c in ["position_start", "position_current"] if c in standings.columns), None)

        if pts_col and pos_col:
            stand_clean = standings[stand_cols_needed + [pts_col, pos_col]].copy()
            stand_clean = stand_clean.rename(columns={
                pts_col: "champ_points_before",
                pos_col: "champ_position_before"
            }).drop_duplicates(["session_key", "driver_number"])
            df = df.merge(stand_clean, on=["session_key", "driver_number"], how="left")

    for col in ["champ_points_before", "champ_position_before"]:
        if col not in df.columns:
            df[col] = np.nan

    # --- Constructor reliability (DNF rate per team per year) ---
    if "team_name" not in df.columns:
        df["team_name"] = "Unknown"
    df["team_name"] = df["team_name"].fillna("Unknown")

    if "year" not in df.columns:
        df["year"] = 2023

    # Build a numeric DNF flag from whichever incident columns are present.
    if "dnf_int" not in df.columns:
        dnf_sources = [col for col in ["dnf", "dsq", "dns"] if col in df.columns]
        if dnf_sources:
            dnf_mask = pd.Series(False, index=df.index)
            for col in dnf_sources:
                dnf_mask = dnf_mask | df[col].fillna(False).astype(bool)
            df["dnf_int"] = dnf_mask.astype(int)
        else:
            df["dnf_int"] = 0

    team_reliability = (
        df.groupby(["year", "team_name"])["dnf_int"]
        .mean()
        .reset_index()
        .rename(columns={"dnf_int": "team_dnf_rate"})
    )
    df = df.merge(team_reliability, on=["year", "team_name"], how="left")
    df["team_dnf_rate"] = df["team_dnf_rate"].fillna(0.05)

    # --- Driver form: rolling avg of last 3 race positions ---
    if "position" not in df.columns:
        df["position"] = 10
    df["position"] = pd.to_numeric(df["position"], errors="coerce").fillna(10)

    if "date_start" not in df.columns:
        df["date_start"] = "2023-01-01"

    df = df.sort_values(["driver_number", "year", "date_start"])
    df["driver_form_3"] = (
        df.groupby("driver_number")["position"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df["driver_form_3"] = df["driver_form_3"].fillna(df["position"])

    # --- Circuit specialist: driver avg finish at this circuit ---
    if "circuit_short_name" not in df.columns:
        df["circuit_short_name"] = "Unknown"
    df["circuit_short_name"] = df["circuit_short_name"].fillna("Unknown")

    df["circuit_specialist_score"] = (
        df.groupby(["driver_number", "circuit_short_name"])["position"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df["circuit_specialist_score"] = df["circuit_specialist_score"].fillna(df["position"])

    # --- Encode categorical: circuit_type, starting_compound ---
    circuit_type_map = {
        "Permanent": 0,
        "Temporary - Street": 1,
        "Temporary - Road": 2,
    }
    df["circuit_type_enc"] = df["circuit_type"].map(circuit_type_map).fillna(0).astype(int)

    compound_map = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4, "UNKNOWN": 2}
    df["starting_compound"] = df["starting_compound"].fillna("MEDIUM").astype(str).str.upper()
    df["compound_enc"] = df["starting_compound"].map(compound_map).fillna(2).astype(int)

    # --- DNF as target modifier: encode DNF as position 21 ---
    for col in ["dnf", "dsq", "dns"]:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False)

    df["finish_position"] = df.apply(
        lambda r: 21 if (r["dnf"] or r["dsq"] or r["dns"]) else r["position"],
        axis=1
    )

    # Final cleanup — fill all numeric NaNs with sensible defaults
    numeric_fill = {
        "grid_position": df.get("position", pd.Series(10)),
        "quali_delta_to_pole": 0.5,
        "total_stints": 2,
        "avg_tyre_age": 2.0,
        "num_pit_stops": 1,
        "avg_stop_duration": 2.5,
        "air_temperature": 25.0,
        "track_temperature": 35.0,
        "humidity": 55.0,
        "wind_speed": 2.0,
        "rainfall": 0,
        "champ_points_before": 50.0,
        "champ_position_before": 10.0,
    }
    for col, default in numeric_fill.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
        else:
            df[col] = default

    print(f"      Feature engineering complete: {len(df)} rows, {len(df.columns)} columns")
    return df


def run_pipeline(years=None, save=True):
    """
    Full pipeline: fetch → process → save.
    Returns final feature DataFrame.
    """
    if years is None:
        years = [2023, 2024, 2025]

    print("=" * 55)
    print("  PITWALL INTEL — Data Pipeline Starting")
    print("=" * 55)

    print(f"\n[1/9] Fetching race sessions for {years}...")
    race_sessions = get_race_sessions(years)
    print(f"      Found {len(race_sessions)} race sessions")

    if race_sessions.empty:
        print("[ERROR] No sessions found. Check API connectivity.")
        return None

    race_session_keys = race_sessions["session_key"].tolist()
    meeting_keys = race_sessions["meeting_key"].tolist()

    print("[2/9] Fetching qualifying sessions...")
    qual_sessions = get_qualifying_sessions(years)
    qual_pairs = []
    for _, row in race_sessions.iterrows():
        match = qual_sessions[qual_sessions["meeting_key"] == row["meeting_key"]]
        if not match.empty:
            qual_pairs.append((row["meeting_key"], match.iloc[0]["session_key"]))

    print(f"      Found {len(qual_pairs)} qualifying sessions")

    print("[3/9] Fetching race results...")
    results = get_session_results(race_session_keys)
    print(f"      Fetched {len(results)} result rows")

    print("[4/9] Fetching starting grids...")
    grids = get_starting_grids(qual_pairs)
    print(f"      Fetched {len(grids)} grid rows")

    print("[5/9] Fetching tyre stints...")
    stints = get_stints(race_session_keys)
    print(f"      Fetched {len(stints)} stint rows")

    print("[6/9] Fetching pit stops...")
    pits = get_pit_stops(race_session_keys)
    print(f"      Fetched {len(pits)} pit stop rows")

    print("[7/9] Fetching weather data...")
    weather = get_weather(race_session_keys)
    print(f"      Fetched weather for {len(weather)} sessions")

    print("[8/9] Fetching championship standings...")
    standings = get_championship_standings(race_session_keys)
    print(f"      Fetched {len(standings)} standing rows")

    print("[8b] Fetching driver info and meetings...")
    drivers = get_drivers(race_session_keys[:10])  # sample for speed, enough for mapping
    meetings = get_meetings(years)

    print("[9/9] Engineering features...")
    features_df = engineer_features(
        race_sessions, results, grids, stints, pits,
        weather, standings, drivers, meetings
    )
    print(f"      Final dataset: {len(features_df)} rows x {len(features_df.columns)} columns")

    if save:
        os.makedirs(RAW_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        race_sessions.to_csv(os.path.join(RAW_DIR, "sessions.csv"), index=False)
        results.to_csv(os.path.join(RAW_DIR, "results.csv"), index=False)
        grids.to_csv(os.path.join(RAW_DIR, "grids.csv"), index=False)
        stints.to_csv(os.path.join(RAW_DIR, "stints.csv"), index=False)
        pits.to_csv(os.path.join(RAW_DIR, "pits.csv"), index=False)
        weather.to_csv(os.path.join(RAW_DIR, "weather.csv"), index=False)
        features_df.to_csv(os.path.join(PROCESSED_DIR, "features_final.csv"), index=False)
        print(f"\n  Data saved to: {PROCESSED_DIR}/features_final.csv")

    print("\n  Pipeline complete.")
    print("=" * 55)
    return features_df


if __name__ == "__main__":
    df = run_pipeline()
    if df is not None:
        print(df[["year", "circuit_short_name", "name_acronym",
                   "grid_position", "finish_position", "team_name"]].head(20))
