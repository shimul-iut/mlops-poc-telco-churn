"""
generate_batch.py
-----------------
Simulates a data collection pipeline by sampling 10% of the training dataset
and optionally injecting realistic drift into the batch.

This mimics what a production system would do: collect incoming inference
requests over a time window and dump them into a CSV for drift analysis.

Usage:
    # No drift — clean sample from training distribution
    python src/generate_batch.py

    # With mild drift
    python src/generate_batch.py --drift mild

    # With moderate drift
    python src/generate_batch.py --drift moderate

    # With severe drift
    python src/generate_batch.py --drift severe

    # Then run drift analysis on the generated batch
    python src/monitor.py data/current_batch.csv
"""

import pandas as pd
import numpy as np
import argparse
import os

SOURCE_DATA_PATH = "data/telco_churn.csv"
OUTPUT_PATH = "data/current_batch.csv"
SAMPLE_FRACTION = 0.10  # 10% of training data

# ---------------------------------------------------------------------------
# Drift profiles: each defines how much to distort specific features.
# These are meant to simulate realistic business shifts, e.g.:
#   - A pricing change pushing MonthlyCharges up
#   - Customer acquisition targeting newer/shorter-tenure customers
#   - A shift toward month-to-month contracts after a promotion
# ---------------------------------------------------------------------------
DRIFT_PROFILES = {
    "none": {},
    "mild": {
        "MonthlyCharges": {"shift": +10.0},       # prices nudged up slightly
        "tenure":         {"shift": -4.0},         # slightly newer customers
    },
    "moderate": {
        "MonthlyCharges": {"shift": +25.0},        # noticeable pricing increase
        "tenure":         {"shift": -10.0},        # meaningfully newer cohort
        "Contract":       {"distribution": [0.60, 0.25, 0.15]},  # more month-to-month
    },
    "severe": {
        "MonthlyCharges": {"shift": +40.0},        # large pricing jump
        "tenure":         {"shift": -18.0},        # mostly very new customers
        "Contract":       {"distribution": [0.80, 0.15, 0.05]},  # dominated by month-to-month
        "TotalCharges":   {"shift": -200.0},       # lower total spend (new customers)
    },
}


def apply_drift(df: pd.DataFrame, profile: dict) -> pd.DataFrame:
    """Apply a drift profile to a copy of the dataframe."""
    df = df.copy()

    for feature, config in profile.items():
        if feature not in df.columns:
            print(f"  Warning: feature '{feature}' not found in data, skipping.")
            continue

        if "shift" in config:
            shift = config["shift"]
            df[feature] = (df[feature] + shift).clip(lower=0)
            print(f"  Shifted '{feature}' by {shift:+.1f}")

        elif "distribution" in config:
            dist = config["distribution"]
            n_categories = df[feature].nunique()
            # Pad or trim distribution to match actual number of categories
            if len(dist) != n_categories:
                dist = dist[:n_categories]
                dist = [p / sum(dist) for p in dist]  # re-normalise
            categories = sorted(df[feature].unique())
            df[feature] = np.random.choice(categories, size=len(df), p=dist)
            print(f"  Re-distributed '{feature}' with weights {dist}")

    return df


def generate_batch(drift_level: str = "none") -> None:
    if not os.path.exists(SOURCE_DATA_PATH):
        raise FileNotFoundError(
            f"Source data not found at '{SOURCE_DATA_PATH}'. "
            "Run prepare_data.py first."
        )

    df = pd.read_csv(SOURCE_DATA_PATH)
    total_rows = len(df)
    sample_size = max(1, int(total_rows * SAMPLE_FRACTION))

    print(f"Source dataset: {total_rows} rows  →  sampling {sample_size} rows ({SAMPLE_FRACTION:.0%})")

    # Stratified sample to preserve churn ratio in the batch
    # Drop the grouping column inside the lambda to avoid the FutureWarning
    # (pandas ≥2.2 will exclude grouping columns by default; this opts in early
    # without relying on `include_groups`, which is missing from the type stubs).
    churn_col = df["Churn"].copy()
    batch = df.drop(columns=["Churn"]).groupby(churn_col, group_keys=False).apply(
        lambda g: g.sample(frac=SAMPLE_FRACTION, random_state=42)
    ).reset_index(drop=True)
    # Restore Churn column
    batch["Churn"] = churn_col.loc[batch.index].values

    print(f"Batch churn rate: {batch['Churn'].mean():.2%}  (reference: {df['Churn'].mean():.2%})")

    # Apply drift profile
    profile = DRIFT_PROFILES.get(drift_level, {})
    if profile:
        print(f"\nApplying '{drift_level}' drift profile:")
        batch = apply_drift(batch, profile)
    else:
        print(f"\nDrift level: none — batch reflects training distribution.")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    batch.to_csv(OUTPUT_PATH, index=False)

    print(f"\nBatch saved to '{OUTPUT_PATH}'  ({len(batch)} rows)")
    print(f"\nNext step — run drift analysis:")
    print(f"  python src/monitor.py {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a synthetic inference batch for drift monitoring."
    )
    parser.add_argument(
        "--drift",
        choices=["none", "mild", "moderate", "severe"],
        default="none",
        help="Drift level to inject into the batch (default: none)",
    )
    args = parser.parse_args()
    generate_batch(drift_level=args.drift)
