"""
Phase 2: Compute Cumulative Sample Loss (CSL).

Reads the per-image, per-epoch loss log produced by train.py
and calculates one CSL score for every image.

CSL = sum of the sample's loss across all training epochs.
"""

import os

import pandas as pd


# ============================================================
# Configuration
# ============================================================

LOSS_LOG_PATH = "results/per_sample_epoch_loss.csv"
OUTPUT_PATH = "results/csl_scores.csv"


# ============================================================
# Load loss log
# ============================================================

loss_df = pd.read_csv(LOSS_LOG_PATH)

print("Loss records:", len(loss_df))
print("Unique images:", loss_df["image_id"].nunique())
print("Unique epochs:", loss_df["epoch"].nunique())


# ============================================================
# Compute CSL
# ============================================================

csl_df = (
    loss_df
    .groupby("image_id", as_index=False)["loss"]
    .sum()
    .rename(columns={"loss": "csl_score"})
)


# ============================================================
# Save CSL scores
# ============================================================

os.makedirs("results", exist_ok=True)

csl_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# Summary
# ============================================================

print()
print("CSL computation complete.")
print("Images:", len(csl_df))
print("Output:", OUTPUT_PATH)
print()
print(csl_df.head())
print()
print("CSL summary:")
print(csl_df["csl_score"].describe())