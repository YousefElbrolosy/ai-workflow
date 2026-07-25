#!/usr/bin/env python3
"""Generate the Part 2 model-comparison figures from
reports/content/model_comparison.json and the retrained overall model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
CONTENT_DIR = ROOT / "reports" / "content"
MODELS_DIR = ROOT / "models"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#2a78d6"
BLUE_DARK = "#184f95"
ORANGE = "#eb6834"
MUTED_FILL = "#c3c2b7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK_PRIMARY, "axes.edgecolor": BASELINE, "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "axes.grid": True,
    "grid.color": GRIDLINE, "grid.linewidth": 1.0, "axes.axisbelow": True,
    "font.size": 10.5, "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlecolor": INK_PRIMARY,
})


def style_axes(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)


def money(x, _pos=None):
    if abs(x) >= 1_000_000:
        return f"${x/1_000_000:,.1f}M"
    if abs(x) >= 1_000:
        return f"${x/1_000:,.0f}K"
    return f"${x:,.0f}"


with open(CONTENT_DIR / "model_comparison.json") as fh:
    C = json.load(fh)

MODEL_LABELS = {
    "naive": "Naive\n(last 28d)", "seasonal_naive": "Seasonal naive\n(prior year)",
    "ridge": "Ridge", "random_forest": "Random\nForest",
    "hist_gbm": "Hist\nGBM", "lightgbm": "LightGBM",
}
LEARNED = ["ridge", "random_forest", "hist_gbm", "lightgbm"]

# ---------------------------------------------------------------------------
# Fig A: avg normalized MAE by learned model family, across all 11 series
# ---------------------------------------------------------------------------
avg_nmae = C["avg_normalized_mae"]
order = sorted(LEARNED, key=lambda m: avg_nmae[m])
winner = C["winner"]

fig, ax = plt.subplots(figsize=(7.5, 5))
colors = [BLUE if m == winner else MUTED_FILL for m in order]
bars = ax.bar([MODEL_LABELS[m] for m in order], [avg_nmae[m] for m in order], color=colors, width=0.55)
for b, m in zip(bars, order):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(avg_nmae.values()) * 0.015,
            f"{avg_nmae[m]:.2f}", ha="center", fontsize=9.5, color=INK_SECONDARY)
ax.set_ylabel("Avg. normalized MAE across 11 series\n(MAE ÷ series mean revenue — lower is better)", fontsize=9.5)
ax.set_title(f"Model comparison — {MODEL_LABELS[winner].replace(chr(10), ' ')} selected")
style_axes(ax)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig09_model_comparison.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig B: per-series MAE, naive vs winner (does the model beat the baseline
# everywhere, not just on average?)
# ---------------------------------------------------------------------------
series_names = list(C["per_series_metrics"].keys())
naive_mae = [C["per_series_metrics"][s]["naive"]["mae"] for s in series_names]
winner_mae = [C["per_series_metrics"][s][winner]["mae"] for s in series_names]
mean_actual = [C["full_series_mean"][s] for s in series_names]
naive_norm = [n / m for n, m in zip(naive_mae, mean_actual)]
winner_norm = [w / m for w, m in zip(winner_mae, mean_actual)]

order2 = np.argsort(mean_actual)[::-1]
series_sorted = [series_names[i] for i in order2]
naive_sorted = [naive_norm[i] for i in order2]
winner_sorted = [winner_norm[i] for i in order2]

fig, ax = plt.subplots(figsize=(9, 5.5))
y = np.arange(len(series_sorted))
h = 0.36
ax.barh(y + h / 2, naive_sorted, height=h, color=MUTED_FILL, label="Naive baseline")
ax.barh(y - h / 2, winner_sorted, height=h, color=BLUE, label=f"{MODEL_LABELS[winner].replace(chr(10), ' ')} (selected)")
ax.set_yticks(y)
ax.set_yticklabels(series_sorted, fontsize=9.5)
ax.set_xlabel("Normalized MAE on held-out period (MAE ÷ series mean revenue)")
ax.set_title("Selected model vs. naive baseline, per series")
ax.legend(frameon=False, loc="lower right", fontsize=9.5)
ax.invert_yaxis()
style_axes(ax)
ax.spines["left"].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig10_model_vs_naive_per_series.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig C: actual vs predicted, overall series holdout
# ---------------------------------------------------------------------------
hold = pd.read_csv(CONTENT_DIR / "overall_holdout_predictions.csv", parse_dates=["date"])
fig, ax = plt.subplots(figsize=(9.5, 4.5))
ax.plot(hold["date"], hold["actual"], color=INK_PRIMARY, linewidth=2, label="Actual")
ax.plot(hold["date"], hold["predicted"], color=BLUE, linewidth=2, linestyle="--", label="Predicted")
ax.fill_between(hold["date"], hold["actual"], hold["predicted"], color=BLUE, alpha=0.08)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(money))
ax.set_title("Overall revenue: actual vs. predicted next-30-day total (held-out period)")
ax.legend(frameon=False, loc="upper left", fontsize=9.5)
style_axes(ax)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig11_actual_vs_predicted.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig D: feature importance, overall model
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(ROOT))
from sklearn.inspection import permutation_importance
from aavail.features import build_supervised

model = joblib.load(MODELS_DIR / "overall.joblib")
meta = json.load(open(MODELS_DIR / "metadata.json"))
cols = meta["series"]["overall"]["feature_columns"]
ts_all = pd.read_csv(ROOT / "data" / "processed" / "ts_all.csv", parse_dates=["date"], index_col="date")
X_all, y_all = build_supervised(ts_all)
perm = permutation_importance(model, X_all[cols], y_all, n_repeats=8, random_state=42, n_jobs=-1)
importances = pd.Series(perm.importances_mean, index=cols).sort_values(ascending=False).head(12)

fig, ax = plt.subplots(figsize=(8, 5.5))
y = np.arange(len(importances))[::-1]
ax.barh(y, importances.values, color=BLUE, height=0.6)
ax.set_yticks(y)
ax.set_yticklabels(importances.index, fontsize=9)
ax.set_title("Top feature importances — overall revenue model")
style_axes(ax)
ax.spines["left"].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig12_feature_importance.png", dpi=200)
plt.close(fig)

print("wrote 4 figures to", FIG_DIR)
