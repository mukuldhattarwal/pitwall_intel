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

BASE_URL = "https://api.openf1.org/v1"
YEARS = [2023, 2024, 2025]


def fetch(endpoint, params=None, retries=3):
    """Generic fetch with retry logic."""
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"  [retry {attempt+1}] {endpoint} — {e}")
            time.sleep(2)
    return []


def get_race_sessions(years=YEARS):
    """Get all race session keys across seasons."""
    sessions = []
    for year in years:
        print(f"Fetching sessions for {year}...")
        data = fetch("sessions", {"year": year, "session_type": "Race", "session_name": "Race"})
        sessions.extend(data)
        time.sleep(0.5)
    df = pd.DataFrame(sessions)
    if df.empty:
        return df
    df = df[df["session_name"] == "Race"].copy()
    return df[["session_key", "meeting_key", "circuit_key", "circuit_short_name",
               "country_name", "circuit_type", "year", "date_start"]].drop_duplicates()


def get_qualifying_sessions(years=YEARS):
    """Get all qualifying session keys."""
    sessions = []
    for year in years:
        data = fetch("sessions", {"year": year, "session_name": "Qualifying"})
        sessions.extend(data)
        time.sleep(0.5)
    df = pd.DataFrame(sessions)
    if df.empty:
        return df
    return df[["session_key", "meeting_key"]].rename(columns={"session_key": "quali_session_key"})


def get_race_results(race_sessions_df):
    """Get final race results for all sessions."""
    all_results = []
    for _, row in race_sessions_df.iterrows():
        skey = row["session_key"]
        data = fetch("session_result", {"session_key": skey})
        for r in data:
            r["session_key"] = skey
            r["meeting_key"] = row["meeting_key"]
            r["circuit_short_name"] = row["circuit_short_name"]
            r["circuit_type"] = row.get("circuit_type", "Permanent")
            r["year"] = row["year"]
            r["date_start"] = row["date_start"]
        all_results.extend(data)
        time.sleep(0.4)
    df = pd.DataFrame(all_results)
    return df


def get_starting_grids(race_sessions_df, quali_sessions_df):
    """Get qualifying times and grid positions."""
    all_grids = []
    merged = race_sessions_df.merge(quali_sessions_df, on="meeting_key", how="left")
    for _, row in merged.iterrows():
        qkey = row.get("quali_session_key")
        if pd.isna(qkey):
            continue
        data = fetch("starting_grid", {"session_key": int(qkey)})
        for r in data:
            r["meeting_key"] = row["meeting_key"]
            r["year"] = row["year"]
        all_grids.extend(data)
        time.sleep(0.4)
    df = pd.DataFrame(all_grids)
    if df.empty:
        return df
    return df[["meeting_key", "year", "driver_number", "position", "lap_duration"]].rename(
        columns={"position": "grid_position", "lap_duration": "quali_time"}
    )


def get_stints(race_sessions_df):
    """Get tyre strategy data."""
    all_stints = []
    for _, row in race_sessions_df.iterrows():
        skey = row["session_key"]
        data = fetch("stints", {"session_key": skey})
        for r in data:
            r["meeting_key"] = row["meeting_key"]
            r["year"] = row["year"]
        all_stints.extend(data)
        time.sleep(0.4)
    return pd.DataFrame(all_stints)


def get_pit_stops(race_sessions_df):
    """Get pit stop data."""
    all_pits = []
    for _, row in race_sessions_df.iterrows():
        skey = row["session_key"]
        data = fetch("pit", {"session_key": skey})
        for r in data:
            r["meeting_key"] = row["meeting_key"]
            r["year"] = row["year"]
        all_pits.extend(data)
        time.sleep(0.4)
    return pd.DataFrame(all_pits)


def get_weather(race_sessions_df):
    """Get average weather per race session."""
    weather_rows = []
    for _, row in race_sessions_df.iterrows():
        skey = row["session_key"]
        data = fetch("weather", {"session_key": skey})
        if data:
            df_w = pd.DataFrame(data)
            avg = {
                "meeting_key": row["meeting_key"],
                "year": row["year"],
                "avg_air_temp": df_w["air_temperature"].mean() if "air_temperature" in df_w else np.nan,
                "avg_track_temp": df_w["track_temperature"].mean() if "track_temperature" in df_w else np.nan,
                "rainfall": int(df_w["rainfall"].max() > 0) if "rainfall" in df_w else 0,
                "avg_wind_speed": df_w["wind_speed"].mean() if "wind_speed" in df_w else np.nan,
            }
            weather_rows.append(avg)
        time.sleep(0.4)
    return pd.DataFrame(weather_rows)


def get_championship_standings(race_sessions_df):
    """Get driver championship standings per race."""
    all_standings = []
    for _, row in race_sessions_df.iterrows():
        skey = row["session_key"]
        data = fetch("championship_drivers", {"session_key": skey})
        for r in data:
            r["meeting_key"] = row["meeting_key"]
            r["year"] = row["year"]
        all_standings.extend(data)
        time.sleep(0.4)
    df = pd.DataFrame(all_standings)
    if df.empty:
        return df
    return df[["meeting_key", "year", "driver_number", "points_start", "position_start"]].rename(
        columns={"points_start": "championship_points_before",
                 "position_start": "championship_pos_before"}
    )


def build_tyre_features(stints_df):
    """Aggregate tyre strategy per driver per race."""
    if stints_df.empty:
        return pd.DataFrame()
    features = []
    for (meeting_key, driver_number), grp in stints_df.groupby(["meeting_key", "driver_number"]):
        compounds = grp["compound"].tolist()
        n_stops = len(grp) - 1
        starting_compound = compounds[0] if compounds else "UNKNOWN"
        compound_map = {"SOFT": 1, "MEDIUM": 2, "HARD": 3, "INTERMEDIATE": 4, "WET": 5}
        starting_compound_enc = compound_map.get(starting_compound, 0)
        tyre_age = grp["tyre_age_at_start"].iloc[0] if "tyre_age_at_start" in grp.columns else 0
        features.append({
            "meeting_key": meeting_key,
            "driver_number": driver_number,
            "n_pit_stops": n_stops,
            "starting_compound_enc": starting_compound_enc,
            "tyre_age_at_start": tyre_age,
        })
    return pd.DataFrame(features)


def build_pit_features(pit_df):
    """Aggregate pit stop performance per driver per race."""
    if pit_df.empty:
        return pd.DataFrame()
    grp = pit_df.groupby(["meeting_key", "driver_number"])
    agg = grp["stop_duration"].mean().reset_index() if "stop_duration" in pit_df.columns else \
          grp["lane_duration"].mean().reset_index()
    agg.columns = ["meeting_key", "driver_number", "avg_pit_duration"]
    return agg


def build_driver_form(results_df):
    """Rolling average finish position over last 3 races per driver."""
    results_df = results_df.sort_values(["driver_number", "year", "date_start"])
    results_df["finish_position"] = pd.to_numeric(results_df["position"], errors="coerce")
    results_df["dnf_flag"] = results_df["dnf"].astype(int) if "dnf" in results_df.columns else 0
    results_df["finish_position_adj"] = results_df.apply(
        lambda r: 21 if r.get("dnf", False) or r.get("dns", False) or r.get("dsq", False) else r["finish_position"],
        axis=1
    )
    results_df["driver_form_score"] = results_df.groupby("driver_number")["finish_position_adj"] \
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    return results_df


def build_circuit_type_encoding(results_df):
    """Encode circuit type as numeric."""
    circuit_map = {"Permanent": 0, "Temporary - Street": 1, "Temporary - Road": 2}
    results_df["circuit_type_enc"] = results_df["circuit_type"].map(circuit_map).fillna(0)
    return results_df


def build_master_dataset(save_path="data/processed/master_features.csv"):
    print("\n=== PitWall Intel — Data Pipeline ===\n")

    print("[1/8] Fetching race sessions...")
    race_sessions = get_race_sessions()
    print(f"  Found {len(race_sessions)} race sessions")

    print("[2/8] Fetching qualifying sessions...")
    quali_sessions = get_qualifying_sessions()

    print("[3/8] Fetching race results...")
    results = get_race_results(race_sessions)
    print(f"  Found {len(results)} driver-race results")

    print("[4/8] Fetching starting grids...")
    grids = get_starting_grids(race_sessions, quali_sessions)

    print("[5/8] Fetching tyre stints...")
    stints = get_stints(race_sessions)

    print("[6/8] Fetching pit stops...")
    pits = get_pit_stops(race_sessions)

    print("[7/8] Fetching weather...")
    weather = get_weather(race_sessions)

    print("[8/8] Fetching championship standings...")
    standings = get_championship_standings(race_sessions)

    print("\nBuilding features...")
    results = build_driver_form(results)
    results = build_circuit_type_encoding(results)

    tyre_feats = build_tyre_features(stints)
    pit_feats = build_pit_features(pits)

    master = results.copy()

    if not grids.empty:
        master = master.merge(grids[["meeting_key", "driver_number", "grid_position", "quali_time"]],
                              on=["meeting_key", "driver_number"], how="left")

    if not tyre_feats.empty:
        master = master.merge(tyre_feats, on=["meeting_key", "driver_number"], how="left")

    if not pit_feats.empty:
        master = master.merge(pit_feats, on=["meeting_key", "driver_number"], how="left")

    if not weather.empty:
        master = master.merge(weather[["meeting_key", "avg_air_temp", "avg_track_temp",
                                       "rainfall", "avg_wind_speed"]],
                              on="meeting_key", how="left")

    if not standings.empty:
        master = master.merge(standings[["meeting_key", "driver_number",
                                         "championship_points_before", "championship_pos_before"]],
                              on=["meeting_key", "driver_number"], how="left")

    # Compute qualifying delta to pole
    if "quali_time" in master.columns:
        pole_times = master.groupby("meeting_key")["quali_time"].min().reset_index()
        pole_times.columns = ["meeting_key", "pole_time"]
        master = master.merge(pole_times, on="meeting_key", how="left")
        master["quali_delta_to_pole"] = master["quali_time"] - master["pole_time"]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    master.to_csv(save_path, index=False)
    print(f"\nMaster dataset saved: {save_path}")
    print(f"Shape: {master.shape}")
    return master


if __name__ == "__main__":
    df = build_master_dataset()
    print(df.head())
