"""
Phase 5: Statistical analysis of CSL vs INT8 compression damage.

Inputs:
    results/damage.csv
    results/csl_scores.csv
    results/per_sample_epoch_loss.csv

Main analysis:
1. Mann-Whitney U: CSL, flipped vs non-flipped
2. Spearman/Pearson: CSL vs confidence_drop
3. AUROC: CSL predicting flipped
4. Flip rate by CSL quartile
5. Repeat 1-3 using final-epoch loss as baseline
6. Save CSL vs confidence-drop scatter plot

Outputs:
    results/analysis_scatter_csl_vs_confidence_drop.png
    results/analysis_merged.csv
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import (
    mannwhitneyu,
    pearsonr,
    spearmanr,
)
from sklearn.metrics import roc_auc_score


# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = "results"

DAMAGE_PATH = os.path.join(
    RESULTS_DIR,
    "damage.csv",
)

CSL_PATH = os.path.join(
    RESULTS_DIR,
    "csl_scores.csv",
)

LOSS_PATH = os.path.join(
    RESULTS_DIR,
    "per_sample_epoch_loss.csv",
)

MERGED_OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "analysis_merged.csv",
)

SCATTER_OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "analysis_scatter_csl_vs_confidence_drop.png",
)


# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("PHASE 5: CSL vs COMPRESSION DAMAGE ANALYSIS")
print("=" * 70)

print()
print("Loading datasets...")

damage = pd.read_csv(DAMAGE_PATH)
csl = pd.read_csv(CSL_PATH)
loss = pd.read_csv(LOSS_PATH)

print(
    f"Damage rows: {len(damage)}"
)

print(
    f"CSL rows: {len(csl)}"
)

print(
    f"Loss rows: {len(loss)}"
)


# ============================================================
# Validate basic structure
# ============================================================

required_damage_columns = [
    "image_id",
    "true_label",
    "original_prediction",
    "original_confidence",
    "quantized_prediction",
    "quantized_confidence",
    "flipped",
    "confidence_drop",
]

required_csl_columns = [
    "image_id",
    "csl_score",
]

required_loss_columns = [
    "image_id",
    "epoch",
    "loss",
]

for column in required_damage_columns:

    if column not in damage.columns:
        raise ValueError(
            f"Missing column in damage.csv: {column}"
        )

for column in required_csl_columns:

    if column not in csl.columns:
        raise ValueError(
            f"Missing column in csl_scores.csv: {column}"
        )

for column in required_loss_columns:

    if column not in loss.columns:
        raise ValueError(
            f"Missing column in per_sample_epoch_loss.csv: {column}"
        )


# ============================================================
# Validate uniqueness
# ============================================================

if damage["image_id"].duplicated().any():

    raise ValueError(
        "damage.csv contains duplicate image_id values."
    )

if csl["image_id"].duplicated().any():

    raise ValueError(
        "csl_scores.csv contains duplicate image_id values."
    )


# ============================================================
# Extract final-epoch loss
# ============================================================

loss["epoch"] = pd.to_numeric(
    loss["epoch"]
)

loss["loss"] = pd.to_numeric(
    loss["loss"]
)

final_epoch = loss["epoch"].max()

final_loss = (
    loss[
        loss["epoch"] == final_epoch
    ][
        [
            "image_id",
            "loss",
        ]
    ]
    .rename(
        columns={
            "loss": "final_epoch_loss"
        }
    )
)

if final_loss["image_id"].duplicated().any():

    raise ValueError(
        "Multiple final-epoch loss records found "
        "for the same image_id."
    )

print()
print(
    f"Final epoch selected: {final_epoch}"
)

print(
    f"Final-epoch loss rows: {len(final_loss)}"
)


# ============================================================
# Merge datasets
# ============================================================

data = damage.merge(
    csl,
    on="image_id",
    how="inner",
    validate="one_to_one",
)

data = data.merge(
    final_loss,
    on="image_id",
    how="inner",
    validate="one_to_one",
)

print()
print(
    f"Merged analysis rows: {len(data)}"
)

if len(data) != 3000:

    raise ValueError(
        f"Expected 3000 merged images, "
        f"but found {len(data)}."
    )


# ============================================================
# Validate missing values
# ============================================================

analysis_columns = [
    "image_id",
    "csl_score",
    "final_epoch_loss",
    "confidence_drop",
    "flipped",
]

missing_counts = (
    data[analysis_columns]
    .isna()
    .sum()
)

if missing_counts.any():

    print()
    print(
        "Missing values detected:"
    )

    print(
        missing_counts[
            missing_counts > 0
        ]
    )

    raise ValueError(
        "Missing values found in analysis data."
    )


# ============================================================
# Prepare groups
# ============================================================

flipped = (
    data["flipped"] == 1
)

non_flipped = (
    data["flipped"] == 0
)

csl_flipped = data.loc[
    flipped,
    "csl_score",
]

csl_non_flipped = data.loc[
    non_flipped,
    "csl_score",
]

loss_flipped = data.loc[
    flipped,
    "final_epoch_loss",
]

loss_non_flipped = data.loc[
    non_flipped,
    "final_epoch_loss",
]


# ============================================================
# 1. Mann-Whitney U: CSL
# ============================================================

print()
print("-" * 70)
print("1. MANN-WHITNEY U TEST: CSL")
print("-" * 70)

csl_u, csl_mw_p = mannwhitneyu(
    csl_flipped,
    csl_non_flipped,
    alternative="two-sided",
)

print(
    f"Flipped images: {len(csl_flipped)}"
)

print(
    f"Non-flipped images: {len(csl_non_flipped)}"
)

print(
    f"Median CSL (flipped): "
    f"{csl_flipped.median():.6f}"
)

print(
    f"Median CSL (non-flipped): "
    f"{csl_non_flipped.median():.6f}"
)

print(
    f"Mann-Whitney U statistic: "
    f"{csl_u:.6f}"
)

print(
    f"Mann-Whitney p-value: "
    f"{csl_mw_p:.12g}"
)

if csl_mw_p <= 0.05:

    print(
        "Result: statistically significant at "
        "alpha = 0.05."
    )

else:

    print(
        "Result: NOT statistically significant "
        "at alpha = 0.05."
    )

    print(
        "The observed difference could plausibly "
        "be noise given this sample."
    )


# ============================================================
# 2. CSL vs confidence_drop correlations
# ============================================================

print()
print("-" * 70)
print("2. CSL vs CONFIDENCE DROP")
print("-" * 70)

pearson_r, pearson_p = pearsonr(
    data["csl_score"],
    data["confidence_drop"],
)

spearman_rho, spearman_p = spearmanr(
    data["csl_score"],
    data["confidence_drop"],
)

print(
    f"Pearson r: {pearson_r:.8f}"
)

print(
    f"Pearson p-value: {pearson_p:.12g}"
)

print(
    f"Spearman rho: {spearman_rho:.8f}"
)

print(
    f"Spearman p-value: {spearman_p:.12g}"
)


# ============================================================
# 3. AUROC: CSL predicting flips
# ============================================================

print()
print("-" * 70)
print("3. AUROC: CSL PREDICTING PREDICTION FLIPS")
print("-" * 70)

y_true = data["flipped"].astype(int)

csl_auc = roc_auc_score(
    y_true,
    data["csl_score"],
)

print(
    f"CSL AUROC: {csl_auc:.8f}"
)

print(
    "Interpretation:"
)

if csl_auc > 0.5:

    print(
        "CSL ranks flipped images above non-flipped "
        "images better than random."
    )

elif csl_auc < 0.5:

    print(
        "CSL ranks flipped images below non-flipped "
        "images on average."
    )

else:

    print(
        "CSL performs at chance level."
    )


# ============================================================
# 4. Flip rate by CSL quartile
# ============================================================

print()
print("-" * 70)
print("4. FLIP RATE BY CSL QUARTILE")
print("-" * 70)

data["csl_quartile"] = pd.qcut(
    data["csl_score"],
    q=4,
    labels=[
        "Bottom 25%",
        "25-50%",
        "50-75%",
        "Top 25%",
    ],
)

quartile_summary = (
    data
    .groupby(
        "csl_quartile",
        observed=True,
    )
    .agg(
        images=("image_id", "count"),
        flips=("flipped", "sum"),
        median_csl=("csl_score", "median"),
    )
)

quartile_summary["flip_rate"] = (
    quartile_summary["flips"]
    / quartile_summary["images"]
    * 100
)

for quartile, row in quartile_summary.iterrows():

    print(
        f"{quartile}: "
        f"{int(row['images'])} images, "
        f"{int(row['flips'])} flips, "
        f"flip rate = "
        f"{row['flip_rate']:.4f}%, "
        f"median CSL = "
        f"{row['median_csl']:.6f}"
    )


# ============================================================
# Monotonicity check
# ============================================================

flip_rates = (
    quartile_summary["flip_rate"]
    .tolist()
)

monotonic_increase = all(
    flip_rates[i] <= flip_rates[i + 1]
    for i in range(
        len(flip_rates) - 1
    )
)

print()

if monotonic_increase:

    print(
        "Quartile pattern: flip rate increases "
        "monotonically with CSL quartile."
    )

else:

    print(
        "Quartile pattern: flip rate does NOT "
        "increase monotonically across all CSL quartiles."
    )


# ============================================================
# 5A. Baseline: Mann-Whitney U using final loss
# ============================================================

print()
print("-" * 70)
print("5A. BASELINE MANN-WHITNEY U: FINAL-EPOCH LOSS")
print("-" * 70)

loss_u, loss_mw_p = mannwhitneyu(
    loss_flipped,
    loss_non_flipped,
    alternative="two-sided",
)

print(
    f"Median final loss (flipped): "
    f"{loss_flipped.median():.8f}"
)

print(
    f"Median final loss (non-flipped): "
    f"{loss_non_flipped.median():.8f}"
)

print(
    f"Mann-Whitney U statistic: "
    f"{loss_u:.6f}"
)

print(
    f"Mann-Whitney p-value: "
    f"{loss_mw_p:.12g}"
)

if loss_mw_p <= 0.05:

    print(
        "Result: statistically significant at "
        "alpha = 0.05."
    )

else:

    print(
        "Result: NOT statistically significant "
        "at alpha = 0.05."
    )

    print(
        "The observed difference could plausibly "
        "be noise given this sample."
    )


# ============================================================
# 5B. Baseline: correlations using final loss
# ============================================================

print()
print("-" * 70)
print("5B. BASELINE CORRELATIONS: FINAL LOSS")
print("-" * 70)

loss_pearson_r, loss_pearson_p = pearsonr(
    data["final_epoch_loss"],
    data["confidence_drop"],
)

loss_spearman_rho, loss_spearman_p = spearmanr(
    data["final_epoch_loss"],
    data["confidence_drop"],
)

print(
    f"Pearson r: {loss_pearson_r:.8f}"
)

print(
    f"Pearson p-value: {loss_pearson_p:.12g}"
)

print(
    f"Spearman rho: {loss_spearman_rho:.8f}"
)

print(
    f"Spearman p-value: {loss_spearman_p:.12g}"
)


# ============================================================
# 5C. Baseline: AUROC using final loss
# ============================================================

print()
print("-" * 70)
print("5C. BASELINE AUROC: FINAL LOSS PREDICTING FLIPS")
print("-" * 70)

loss_auc = roc_auc_score(
    y_true,
    data["final_epoch_loss"],
)

print(
    f"Final-epoch loss AUROC: "
    f"{loss_auc:.8f}"
)


# ============================================================
# Baseline comparison
# ============================================================

print()
print("-" * 70)
print("CSL vs FINAL-LOSS BASELINE")
print("-" * 70)

print(
    f"CSL AUROC:              {csl_auc:.8f}"
)

print(
    f"Final-loss AUROC:       {loss_auc:.8f}"
)

print(
    f"CSL Spearman rho:       {spearman_rho:.8f}"
)

print(
    f"Final-loss Spearman:    {loss_spearman_rho:.8f}"
)


# ============================================================
# 6. Scatter plot
# ============================================================

print()
print("-" * 70)
print("6. SCATTER PLOT")
print("-" * 70)

plt.figure(
    figsize=(10, 7)
)

non_flipped_plot = data[
    data["flipped"] == 0
]

flipped_plot = data[
    data["flipped"] == 1
]

plt.scatter(
    non_flipped_plot["csl_score"],
    non_flipped_plot["confidence_drop"],
    label="Not flipped",
    alpha=0.45,
    s=18,
)

plt.scatter(
    flipped_plot["csl_score"],
    flipped_plot["confidence_drop"],
    label="Flipped",
    alpha=0.75,
    s=24,
)

plt.axhline(
    0,
    linewidth=1,
)

plt.xlabel(
    "CSL score"
)

plt.ylabel(
    "Confidence drop "
    "(FP32 true-label confidence − INT8 true-label confidence)"
)

plt.title(
    "CSL vs INT8 Compression Damage"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    SCATTER_OUTPUT_PATH,
    dpi=300,
)

plt.close()

print(
    "Scatter plot saved:",
    SCATTER_OUTPUT_PATH,
)


# ============================================================
# Save merged analysis dataset
# ============================================================

data.to_csv(
    MERGED_OUTPUT_PATH,
    index=False,
)

print(
    "Merged analysis data saved:",
    MERGED_OUTPUT_PATH,
)


# ============================================================
# Final research summary
# ============================================================

print()
print("=" * 70)
print("PHASE 5 ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"Images analyzed: {len(data)}"
)

print(
    f"Prediction flips: {int(data['flipped'].sum())}"
)

print(
    f"Overall flip rate: "
    f"{data['flipped'].mean() * 100:.4f}%"
)

print()
print(
    f"CSL Mann-Whitney p-value: "
    f"{csl_mw_p:.12g}"
)

print(
    f"CSL Pearson r: "
    f"{pearson_r:.8f}"
)

print(
    f"CSL Spearman rho: "
    f"{spearman_rho:.8f}"
)

print(
    f"CSL AUROC: "
    f"{csl_auc:.8f}"
)

print()
print(
    f"Final-loss Mann-Whitney p-value: "
    f"{loss_mw_p:.12g}"
)

print(
    f"Final-loss Pearson r: "
    f"{loss_pearson_r:.8f}"
)

print(
    f"Final-loss Spearman rho: "
    f"{loss_spearman_rho:.8f}"
)

print(
    f"Final-loss AUROC: "
    f"{loss_auc:.8f}"
)

print()
print(
    "Outputs:"
)

print(
    MERGED_OUTPUT_PATH
)

print(
    SCATTER_OUTPUT_PATH
)

print("=" * 70)