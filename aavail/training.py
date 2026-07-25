"""
Retraining used by the deployed API's /train endpoint (and by the
production-simulation script). This does *not* re-run the Part 2 model
comparison — that selection (Histogram Gradient Boosting, with its tuned
hyperparameters) was already made offline. This module simply refits that
chosen approach on whatever data is currently on disk, which is what lets
the API retrain "at regular intervals with little overhead," per the case
study's suggestion of pointing the trainer at a directory of files.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import joblib
import pandas as pd

from .features import build_supervised
from .ingestion import IngestionError, convert_to_ts, fetch_data, top_countries_by_revenue
from .modeling import make_model
from .monitoring import training_feature_summary

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = ROOT / "models"

DEFAULT_MODEL_FAMILY = "hist_gbm"
DEFAULT_HYPERPARAMETERS = {"max_depth": 3, "learning_rate": 0.1, "max_iter": 100}


def _load_selected_approach(models_dir: Path) -> tuple[str, dict]:
    """Use the model family/hyperparameters chosen in Part 2 if that
    metadata is present; otherwise fall back to the recorded default so a
    from-scratch retrain still works."""
    meta_path = models_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as fh:
            meta = json.load(fh)
        return meta.get("model_family", DEFAULT_MODEL_FAMILY), meta.get("hyperparameters", DEFAULT_HYPERPARAMETERS)
    return DEFAULT_MODEL_FAMILY, DEFAULT_HYPERPARAMETERS


def retrain(data_dirs: Iterable[str], models_dir: Path = DEFAULT_MODELS_DIR,
            top_n: int = 10, as_of: Optional[str] = None) -> dict:
    """Ingest every file in `data_dirs`, rebuild the overall + top-N-country
    daily series, and refit one model per series using the already-selected
    approach. Persists models/metadata.json/training_summary.json exactly as
    Part 2 did, so the API and offline scripts stay interchangeable.

    `as_of`, if given, truncates every series to dates <= as_of before
    training — used by the production simulation to retrain only on data
    that would actually have been available on a given day.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_family, hyperparameters = _load_selected_approach(models_dir)

    frames = []
    for data_dir in data_dirs:
        df = fetch_data(data_dir)
        frames.append(df)
    if not frames:
        raise IngestionError("no data directories provided to retrain()")
    transactions = pd.concat(frames, ignore_index=True, sort=False)
    if as_of is not None:
        transactions = transactions[transactions["date"] <= pd.Timestamp(as_of)]
    if transactions.empty:
        raise IngestionError("no transactions available to train on (check as_of / data_dirs)")

    top_countries = top_countries_by_revenue(transactions, n=top_n)
    series = {"overall": convert_to_ts(transactions)}
    for country in top_countries:
        series[country] = convert_to_ts(transactions, country=country)

    metadata = {
        "model_family": model_family,
        "hyperparameters": hyperparameters,
        "horizon_days": 30,
        "top_n_countries": top_n,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "as_of": as_of,
        "data_dirs": list(data_dirs),
        "series": {},
    }
    trained_series = []
    skipped_series = []
    for name, ts in series.items():
        X, y = build_supervised(ts)
        if len(X) < 20:
            skipped_series.append({"name": name, "reason": f"only {len(X)} usable training rows"})
            continue
        model = make_model(model_family, **hyperparameters)
        model.fit(X, y)
        slug = name.lower().replace(" ", "_")
        joblib.dump(model, models_dir / f"{slug}.joblib")
        metadata["series"][name] = {
            "slug": slug,
            "n_training_rows": len(X),
            "date_range": [str(X.index.min().date()), str(X.index.max().date())],
            "feature_columns": list(X.columns),
            "train_feature_summary": training_feature_summary(X),
        }
        trained_series.append(name)

    with open(models_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)

    return {
        "trained_at": metadata["trained_at"],
        "model_family": model_family,
        "hyperparameters": hyperparameters,
        "series_trained": trained_series,
        "series_skipped": skipped_series,
        "n_transactions": int(len(transactions)),
    }
