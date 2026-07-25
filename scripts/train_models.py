#!/usr/bin/env python3
"""
Part 2: compare candidate 30-day revenue forecasting models, select and
tune the best approach, then retrain it on all available data for each
target series (overall + the ten highest-revenue countries) and persist
the deployable models.

Usage: python scripts/train_models.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aavail.features import build_supervised, latest_feature_row
from aavail.modeling import (
    ALL_MODEL_NAMES,
    LEARNED_MODEL_NAMES,
    make_model,
    naive_predict,
    regression_metrics,
    seasonal_naive_predict,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
CONTENT_DIR = ROOT / "reports" / "content"
MODELS_DIR.mkdir(exist_ok=True)
CONTENT_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_FRACTION = 0.2
MIN_HOLDOUT_ROWS = 40

# Small, deliberately cheap hyperparameter grids — tuned once on the overall
# (highest-data-volume) series via time-series cross-validation, then reused
# for every series so the comparison stays apples-to-apples.
HP_GRID = {
    "random_forest": [
        {"n_estimators": 200, "max_depth": 4},
        {"n_estimators": 400, "max_depth": 6},
        {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 3},
    ],
    "hist_gbm": [
        {"max_depth": 3, "learning_rate": 0.05, "max_iter": 200},
        {"max_depth": 3, "learning_rate": 0.1, "max_iter": 100},
        {"max_depth": 5, "learning_rate": 0.05, "max_iter": 150},
    ],
    "lightgbm": [
        {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.05, "num_leaves": 15},
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.03, "num_leaves": 31},
        {"n_estimators": 150, "max_depth": -1, "learning_rate": 0.1, "num_leaves": 15},
    ],
    "ridge": [{"alpha": 0.1}, {"alpha": 1.0}, {"alpha": 10.0}],
}


def load_series() -> dict[str, pd.DataFrame]:
    ts_all = pd.read_csv(PROCESSED / "ts_all.csv", parse_dates=["date"], index_col="date")
    ts_top = pd.read_csv(PROCESSED / "ts_top_countries.csv", parse_dates=["date"])
    series = {"overall": ts_all}
    for country in ts_top["country"].unique():
        sub = ts_top[ts_top["country"] == country].set_index("date").drop(columns=["country"])
        series[country] = sub
    return series


def tune_hyperparameters(X: pd.DataFrame, y: pd.Series) -> dict:
    """Time-series cross-validated grid search, run once on the series with
    the most data (overall). Returns the best hyperparameter set per
    learned-model family."""
    tscv = TimeSeriesSplit(n_splits=4)
    best = {}
    for name in ("random_forest", "hist_gbm", "lightgbm", "ridge"):
        best_score, best_hp = np.inf, HP_GRID[name][0]
        for hp in HP_GRID[name]:
            fold_maes = []
            for train_idx, val_idx in tscv.split(X):
                model = make_model(name, **hp)
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                pred = model.predict(X.iloc[val_idx])
                fold_maes.append(regression_metrics(y.iloc[val_idx], pred)["mae"])
            score = float(np.mean(fold_maes))
            if score < best_score:
                best_score, best_hp = score, hp
        best[name] = best_hp
        print(f"  tuned {name}: {best_hp} (cv mae={best_score:,.0f})")
    return best


def evaluate_all_models(X: pd.DataFrame, y: pd.Series, best_hp: dict, test_size: int) -> dict:
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    results = {}
    results["naive"] = regression_metrics(y_test, naive_predict(X_test))
    results["seasonal_naive"] = regression_metrics(y_test, seasonal_naive_predict(X_test))
    for name in LEARNED_MODEL_NAMES:
        model = make_model(name, **best_hp[name])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = regression_metrics(y_test, pred)
    return results, (y_test, X_test)


def main() -> int:
    series = load_series()
    built = {name: build_supervised(ts) for name, ts in series.items()}

    print("Tuning hyperparameters on the overall series (most history)...")
    X_overall, y_overall = built["overall"]
    best_hp = tune_hyperparameters(X_overall, y_overall)

    print("\nEvaluating all candidate models on a held-out tail per series...")
    all_metrics = {}
    holdout_predictions = {}
    for name, (X, y) in built.items():
        test_size = max(MIN_HOLDOUT_ROWS, int(len(X) * HOLDOUT_FRACTION))
        test_size = min(test_size, len(X) // 2)  # never eat more than half a short series
        metrics, (y_test, X_test) = evaluate_all_models(X, y, best_hp, test_size)
        all_metrics[name] = metrics
        if name == "overall":
            best_learned_preview = min(LEARNED_MODEL_NAMES, key=lambda m: metrics[m]["mae"])
            model = make_model(best_learned_preview, **best_hp[best_learned_preview])
            model.fit(X.iloc[:-test_size], y.iloc[:-test_size])
            holdout_predictions["overall"] = pd.DataFrame({
                "date": X_test.index, "actual": y_test.values, "predicted": model.predict(X_test),
            })
        print(f"  {name:16s} " + " | ".join(f"{m}: mae={metrics[m]['mae']:,.0f}" for m in ALL_MODEL_NAMES))

    # ---- pick the winning model family: lowest MAE normalized by each
    # series' full-history mean target, averaged across all 11 series (fair
    # comparison across series of very different scale; UK dwarfs Singapore).
    # Normalizing by the *holdout* mean instead of the full-history mean
    # blows up for lumpy series whose short holdout window happens to
    # average near zero (Singapore) even when the absolute error is small,
    # so the full-history mean is used as a stable denominator.
    full_series_mean = {name: float(y.mean()) for name, (X, y) in built.items()}
    normalized_mae = {m: [] for m in LEARNED_MODEL_NAMES}
    for name, metrics in all_metrics.items():
        for m in LEARNED_MODEL_NAMES:
            normalized_mae[m].append(metrics[m]["mae"] / full_series_mean[name])
    avg_normalized_mae = {m: float(np.mean(v)) for m, v in normalized_mae.items()}
    winner = min(avg_normalized_mae, key=avg_normalized_mae.get)
    print(f"\nSelected model family: {winner} (avg normalized MAE = {avg_normalized_mae[winner]:.3f})")
    print("avg normalized MAE by model:", {k: round(v, 3) for k, v in avg_normalized_mae.items()})

    # ---- retrain the winner on ALL available data per series, for deployment ----
    print(f"\nRetraining '{winner}' on the full history of each series...")
    metadata = {"model_family": winner, "hyperparameters": best_hp[winner],
                "horizon_days": 30, "series": {}}
    for name, (X, y) in built.items():
        model = make_model(winner, **best_hp[winner])
        model.fit(X, y)
        slug = name.lower().replace(" ", "_")
        joblib.dump(model, MODELS_DIR / f"{slug}.joblib")
        metadata["series"][name] = {
            "slug": slug,
            "n_training_rows": len(X),
            "date_range": [str(X.index.min().date()), str(X.index.max().date())],
            "feature_columns": list(X.columns),
        }
    with open(MODELS_DIR / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"wrote {len(built)} model(s) to {MODELS_DIR}")

    # ---- persist comparison artifacts for the report ----
    with open(CONTENT_DIR / "model_comparison.json", "w") as fh:
        json.dump({
            "per_series_metrics": all_metrics,
            "avg_normalized_mae": avg_normalized_mae,
            "full_series_mean": full_series_mean,
            "winner": winner,
            "best_hyperparameters": best_hp,
        }, fh, indent=2)
    holdout_predictions["overall"].to_csv(CONTENT_DIR / "overall_holdout_predictions.csv", index=False)
    print(f"wrote comparison metrics to {CONTENT_DIR / 'model_comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
