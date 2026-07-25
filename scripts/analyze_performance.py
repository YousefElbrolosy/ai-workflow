#!/usr/bin/env python3
"""
Post-production analysis: compare the API's logged predictions against
revenue that has since actually been observed, and look at how prediction
error relates to the business metric (revenue level) itself.

Reads the /predict log entries written by scripts/simulate_queries.py (via
aavail.logger), recomputes the *actual* next-30-day revenue for each
(country, as_of_date) pair from the full now-fully-known history
(cs-train + cs-production), and produces:
  * a time-series plot of predicted vs. known 30-day revenue (overall + a
    couple of representative countries)
  * a plot of prediction error vs. revenue level, to see whether errors
    concentrate at high- or low-revenue points in the business cycle

Usage: python scripts/analyze_performance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aavail.ingestion import convert_to_ts, fetch_data
from aavail.logger import query_logs

FIG_DIR = ROOT / "reports" / "figures"
CONTENT_DIR = ROOT / "reports" / "content"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CONTENT_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
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


def load_predictions() -> pd.DataFrame:
    records = query_logs(kind="predict")
    if not records:
        raise SystemExit("no predict log entries found — run scripts/simulate_queries.py against a running API first")
    df = pd.DataFrame(records)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    return df


def actual_next_30d(ts: pd.DataFrame, as_of_date: pd.Timestamp, horizon: int = 30):
    window = ts.loc[as_of_date + pd.Timedelta(days=1): as_of_date + pd.Timedelta(days=horizon), "revenue"]
    if len(window) < horizon:
        return np.nan  # future not fully observed yet
    return float(window.sum())


def main() -> int:
    preds = load_predictions()
    print(f"loaded {len(preds)} predict log entries spanning "
          f"{preds['as_of_date'].min().date()} .. {preds['as_of_date'].max().date()}")

    transactions = pd.concat([fetch_data("cs-train"), fetch_data("cs-production")], ignore_index=True, sort=False)
    ts_by_country = {"overall": convert_to_ts(transactions)}
    for country in preds["country"].unique():
        if country != "overall":
            ts_by_country[country] = convert_to_ts(transactions, country=country)

    preds["actual_revenue_next_30d"] = preds.apply(
        lambda r: actual_next_30d(ts_by_country[r["country"]], r["as_of_date"]), axis=1
    )
    evaluated = preds.dropna(subset=["actual_revenue_next_30d"]).copy()
    evaluated["error"] = evaluated["predicted_revenue_next_30d"] - evaluated["actual_revenue_next_30d"]
    evaluated["abs_pct_error"] = (evaluated["error"].abs() / evaluated["actual_revenue_next_30d"].clip(lower=1)) * 100

    evaluated.sort_values(["country", "as_of_date"]).to_csv(CONTENT_DIR / "post_production_evaluation.csv", index=False)
    print(f"{len(evaluated)} of {len(preds)} predictions have a fully-observed actual (rest are too near the log's end)")

    summary = (
        evaluated.groupby("country")
        .agg(n=("error", "size"), mae=("error", lambda s: s.abs().mean()),
             mape=("abs_pct_error", "mean"), mean_actual=("actual_revenue_next_30d", "mean"))
        .sort_values("mean_actual", ascending=False)
    )
    with open(CONTENT_DIR / "post_production_summary.json", "w") as fh:
        json.dump(json.loads(summary.reset_index().to_json(orient="records")), fh, indent=2)
    print(summary.round(2))

    # ---- Fig: predicted vs. known, overall series ----
    overall = evaluated[evaluated["country"] == "overall"].sort_values("as_of_date")
    if len(overall):
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(overall["as_of_date"], overall["actual_revenue_next_30d"], color=INK_PRIMARY, linewidth=2, label="Known (actual)")
        ax.plot(overall["as_of_date"], overall["predicted_revenue_next_30d"], color=BLUE, linewidth=2, linestyle="--", label="Predicted")
        ax.fill_between(overall["as_of_date"], overall["actual_revenue_next_30d"], overall["predicted_revenue_next_30d"], color=BLUE, alpha=0.10)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(money))
        ax.set_title("Overall: logged predictions vs. known next-30-day revenue")
        ax.legend(frameon=False, loc="upper right", fontsize=9.5)
        style_axes(ax)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig14_production_predicted_vs_known.png", dpi=200)
        plt.close(fig)

    # ---- Fig: error vs. business metric (revenue level) ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    is_overall = evaluated["country"] == "overall"
    ax.scatter(evaluated.loc[~is_overall, "actual_revenue_next_30d"], evaluated.loc[~is_overall, "error"],
               s=18, color=BASELINE, alpha=0.5, label="Countries")
    ax.scatter(evaluated.loc[is_overall, "actual_revenue_next_30d"], evaluated.loc[is_overall, "error"],
               s=22, color=BLUE, alpha=0.7, label="Overall")
    ax.axhline(0, color=BASELINE, linewidth=1.2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(money))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(money))
    ax.set_xlabel("Known next-30-day revenue (the business metric)")
    ax.set_ylabel("Prediction error (predicted − known)")
    ax.set_title("Does error grow with revenue level?")
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig15_error_vs_business_metric.png", dpi=200)
    plt.close(fig)

    corr = float(evaluated["actual_revenue_next_30d"].corr(evaluated["error"].abs()))
    print(f"correlation(|error|, known revenue) = {corr:.3f}")
    with open(CONTENT_DIR / "post_production_stats.json", "w") as fh:
        json.dump({
            "n_evaluated": int(len(evaluated)),
            "n_logged": int(len(preds)),
            "corr_abs_error_vs_revenue": corr,
            "overall_mae": float(evaluated.loc[is_overall, "error"].abs().mean()) if is_overall.any() else None,
            "overall_mape": float(evaluated.loc[is_overall, "abs_pct_error"].mean()) if is_overall.any() else None,
        }, fh, indent=2)

    print("wrote figures to", FIG_DIR, "and stats to", CONTENT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
