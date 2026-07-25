"""
AAVAIL revenue-forecast API.

Endpoints
---------
GET  /health                          liveness check
POST /train   {data_dirs?, top_n?}    retrain all series on all files in the given directories
POST /predict {country, date}         30-day revenue forecast as of `date`, for `country`
GET  /drift?country=&date=            feature-drift check, independent of /predict
GET  /logs?type=&start=&end=&limit=   read back logged train/predict events

Model state (trained models + per-series metadata) is cached in memory after
first load and invalidated by /train, so repeated /predict calls don't pay
ingestion/deserialization cost on every request — the case study explicitly
says this API only needs to serve a handful of active users, so a single-
process in-memory cache is deliberately chosen over a heavier store.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aavail.features import feature_row_for_date
from aavail.ingestion import IngestionError, convert_to_ts, fetch_data
from aavail.logger import log_event, query_logs
from aavail.monitoring import check_drift
from aavail.training import retrain as retrain_models

MODELS_DIR = ROOT / "models"
DEFAULT_DATA_DIRS = [d.strip() for d in os.environ.get("AAVAIL_DATA_DIRS", "cs-train").split(",") if d.strip()]

app = Flask(__name__)

# ---------------------------------------------------------------------------
# In-memory state: metadata, loaded models, ingested/aggregated series.
# Invalidated wholesale on every successful /train.
# ---------------------------------------------------------------------------
_state = {"metadata": None, "models": {}, "transactions": None, "transactions_data_dirs": None, "ts_cache": {}}


def _load_metadata() -> dict:
    if _state["metadata"] is None:
        meta_path = MODELS_DIR / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError("no trained model found — call POST /train first")
        with open(meta_path) as fh:
            _state["metadata"] = json.load(fh)
    return _state["metadata"]


def _get_model(slug: str):
    if slug not in _state["models"]:
        path = MODELS_DIR / f"{slug}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"no trained model file for '{slug}'")
        _state["models"][slug] = joblib.load(path)
    return _state["models"][slug]


def _get_transactions(data_dirs: list[str]) -> pd.DataFrame:
    if _state["transactions"] is None or _state["transactions_data_dirs"] != data_dirs:
        frames = [fetch_data(d) for d in data_dirs]
        _state["transactions"] = pd.concat(frames, ignore_index=True, sort=False)
        _state["transactions_data_dirs"] = data_dirs
        _state["ts_cache"] = {}
    return _state["transactions"]


def _get_ts(country_key: str, data_dirs: list[str]) -> pd.DataFrame:
    if country_key not in _state["ts_cache"]:
        transactions = _get_transactions(data_dirs)
        country_arg = None if country_key == "overall" else country_key
        _state["ts_cache"][country_key] = convert_to_ts(transactions, country=country_arg)
    return _state["ts_cache"][country_key]


def _resolve_country(requested: str, series_meta: dict) -> str:
    """Case-insensitive match of the requested country against the trained
    series names ('overall' plus the top-N countries from the last train)."""
    lookup = {k.lower(): k for k in series_meta.keys()}
    key = lookup.get((requested or "").strip().lower())
    if key is None:
        raise KeyError(
            f"unknown country '{requested}'. Valid options: {sorted(series_meta.keys())}"
        )
    return key


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/train", methods=["POST"])
def train():
    body = request.get_json(silent=True) or {}
    data_dirs = body.get("data_dirs", DEFAULT_DATA_DIRS)
    top_n = int(body.get("top_n", 10))
    as_of = body.get("as_of")

    started = time.time()
    try:
        result = retrain_models(data_dirs, models_dir=MODELS_DIR, top_n=top_n, as_of=as_of)
    except IngestionError as exc:
        return jsonify({"error": str(exc)}), 400
    duration = time.time() - started

    # invalidate all in-memory caches so the next /predict picks up the new model
    _state["metadata"] = None
    _state["models"] = {}
    _state["transactions"] = None
    _state["transactions_data_dirs"] = None
    _state["ts_cache"] = {}

    record = log_event("train", {"request": body, "result": result, "duration_seconds": round(duration, 3)})
    return jsonify(record), 200


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True) or {}
    country_req = body.get("country")
    date_req = body.get("date")
    if not country_req or not date_req:
        return jsonify({"error": "both 'country' and 'date' (YYYY-MM-DD) are required"}), 400

    started = time.time()
    try:
        metadata = _load_metadata()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    try:
        country_key = _resolve_country(country_req, metadata["series"])
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 400

    series_meta = metadata["series"][country_key]
    data_dirs = metadata.get("data_dirs", DEFAULT_DATA_DIRS)

    try:
        ts = _get_ts(country_key, data_dirs)
        row = feature_row_for_date(ts, date_req, horizon=metadata.get("horizon_days", 30))
    except (ValueError, IngestionError) as exc:
        return jsonify({"error": str(exc)}), 400

    feature_cols = series_meta["feature_columns"]
    if row[feature_cols].isna().any(axis=None):
        return jsonify({
            "error": f"insufficient trailing history to predict as of {date_req} for '{country_key}' "
                     f"(need at least 70 days of prior data)"
        }), 400

    model = _get_model(series_meta["slug"])
    prediction = float(model.predict(row[feature_cols])[0])

    drift = None
    if "train_feature_summary" in series_meta:
        drift = check_drift(series_meta["train_feature_summary"], row[feature_cols])

    duration = time.time() - started
    record = log_event("predict", {
        "country": country_key,
        "as_of_date": date_req,
        "horizon_days": metadata.get("horizon_days", 30),
        "predicted_revenue_next_30d": round(prediction, 2),
        "model_family": metadata["model_family"],
        "model_trained_at": metadata.get("trained_at"),
        "drift_detected": drift["drift_detected"] if drift else None,
        "duration_seconds": round(duration, 3),
    })
    return jsonify(record), 200


@app.route("/drift", methods=["GET"])
def drift_check():
    country_req = request.args.get("country")
    date_req = request.args.get("date")
    if not country_req or not date_req:
        return jsonify({"error": "both 'country' and 'date' query params are required"}), 400

    try:
        metadata = _load_metadata()
        country_key = _resolve_country(country_req, metadata["series"])
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 400

    series_meta = metadata["series"][country_key]
    if "train_feature_summary" not in series_meta:
        return jsonify({"error": "no training feature summary available for drift comparison"}), 503

    try:
        ts = _get_ts(country_key, metadata.get("data_dirs", DEFAULT_DATA_DIRS))
        row = feature_row_for_date(ts, date_req, horizon=metadata.get("horizon_days", 30))
    except (ValueError, IngestionError) as exc:
        return jsonify({"error": str(exc)}), 400

    result = check_drift(series_meta["train_feature_summary"], row[series_meta["feature_columns"]])
    result.update({"country": country_key, "as_of_date": date_req})
    log_event("drift", result)
    return jsonify(result), 200


@app.route("/logs", methods=["GET"])
def logs():
    kind = request.args.get("type")
    start = request.args.get("start")
    end = request.args.get("end")
    limit = request.args.get("limit", type=int)
    return jsonify(query_logs(kind=kind, start=start, end=end, limit=limit)), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
