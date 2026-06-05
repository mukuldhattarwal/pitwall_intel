#!/usr/bin/env python3
"""
PitWall Intel — Quick Launch Script
Run this to set up and launch the full app.
"""

import os
import subprocess
import sys


def run(cmd, label):
    print(f"\n[{label}]")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def main():
    print("=" * 50)
    print("  PITWALL INTEL — LAUNCH SEQUENCE")
    print("=" * 50)

    # Check if data exists
    data_path = "data/processed/master_features.csv"
    model_path = "models/xgb_model.pkl"

    if not os.path.exists(data_path):
        print("\nNo training data found. Running data pipeline...")
        print("This will take 5-10 minutes (API calls to OpenF1).\n")
        ok = run("python data_pipeline.py", "DATA PIPELINE")
        if not ok:
            print("Data pipeline failed. Check your internet connection.")
            sys.exit(1)

    if not os.path.exists(model_path):
        print("\nNo model found. Training now...")
        ok = run("python train_model.py", "MODEL TRAINING")
        if not ok:
            print("Training failed.")
            sys.exit(1)

    print("\nLaunching PitWall Intel...")
    run("streamlit run app/main.py", "STREAMLIT")


if __name__ == "__main__":
    main()
