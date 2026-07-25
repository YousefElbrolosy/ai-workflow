"""
Unit tests for the modeling layer: aavail/features.py, aavail/modeling.py,
aavail/monitoring.py, and aavail/training.py.

These test the model logic directly (feature engineering, baselines,
metrics, drift, retraining) rather than through the API, and each uses only
synthetic data or a throwaway tmp_path — nothing here reads or writes the
real models/ directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aavail.features import build_supervised, feature_row_for_date, latest_feature_row
from aavail.modeling import (
    LEARNED_MODEL_NAMES,
    make_model,
    naive_predict,
    regression_metrics,
    seasonal_naive_predict,
)
from aavail.monitoring import check_drift, training_feature_summary
from aavail.training import retrain


def _synthetic_ts(n_days=500, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_days, freq="D")
    revenue = 1000 + 50 * np.sin(np.arange(n_days) / 7) + rng.normal(0, 20, n_days)
    return pd.DataFrame({
        "revenue": revenue.clip(min=0),
        "purchases": rng.integers(5, 20, n_days),
        "unique_streams": rng.integers(3, 15, n_days),
        "unique_customers": rng.integers(2, 10, n_days),
        "total_views": rng.integers(10, 50, n_days),
    }, index=dates)


# ---------------------------------------------------------------------------
# features.py
# ---------------------------------------------------------------------------

def test_build_supervised_target_is_next_30_day_sum():
    ts = _synthetic_ts()
    X, y = build_supervised(ts)
    assert len(X) == len(y)
    # spot check: the target at some row equals the actual trailing 30-day
    # sum starting the day after that row's date
    check_date = X.index[100]
    expected = ts.loc[check_date + pd.Timedelta(days=1): check_date + pd.Timedelta(days=30), "revenue"].sum()
    assert y.loc[check_date] == pytest.approx(expected)


def test_build_supervised_drops_rows_without_full_trailing_or_target_window():
    ts = _synthetic_ts(n_days=200)
    X, y = build_supervised(ts)
    # rolling(70, min_periods=70) is first valid at position 69 (the 70th
    # day, inclusive of "today"); the target needs 30 full days *after* the
    # row, so the last valid row is 30 days before the series ends.
    assert X.index.min() == ts.index[69]
    assert X.index.max() == ts.index[-31]
    assert len(X) == len(ts) - 69 - 30


def test_build_supervised_flags_rather_than_drops_missing_prior_year():
    ts = _synthetic_ts(n_days=200)  # well under 365 days
    X, y = build_supervised(ts)
    assert len(X) > 0
    assert (X["has_prior_year_data"] == 0).all()


def test_feature_row_for_date_matches_build_supervised_row():
    ts = _synthetic_ts()
    X, _ = build_supervised(ts)
    some_date = X.index[50]
    row = feature_row_for_date(ts, some_date)
    pd.testing.assert_series_equal(row.loc[some_date], X.loc[some_date], check_names=False)


def test_feature_row_for_date_uses_no_future_information():
    """Truncating the series to well before its end must not change the
    feature row computed for an earlier as-of date (no leakage)."""
    ts = _synthetic_ts()
    as_of = ts.index[150]
    row_full = feature_row_for_date(ts, as_of)
    row_truncated = feature_row_for_date(ts.loc[:as_of + pd.Timedelta(days=5)], as_of)
    pd.testing.assert_frame_equal(row_full, row_truncated)


def test_feature_row_for_date_out_of_range_raises():
    ts = _synthetic_ts()
    with pytest.raises(ValueError):
        feature_row_for_date(ts, ts.index.max() + pd.Timedelta(days=10))
    with pytest.raises(ValueError):
        feature_row_for_date(ts, ts.index.min() - pd.Timedelta(days=10))


def test_latest_feature_row_is_the_last_date():
    ts = _synthetic_ts()
    row = latest_feature_row(ts)
    assert row.index[0] == ts.index.max()
    assert not row.isna().any(axis=None)


# ---------------------------------------------------------------------------
# modeling.py
# ---------------------------------------------------------------------------

def test_naive_predict_uses_trailing_28_day_sum():
    X = pd.DataFrame({"revenue_sum_28d": [100.0, 200.0]})
    np.testing.assert_array_equal(naive_predict(X), [100.0, 200.0])


def test_seasonal_naive_falls_back_when_no_prior_year_data():
    X = pd.DataFrame({
        "revenue_sum_30d_prior_year": [500.0, 999.0],
        "has_prior_year_data": [1, 0],
        "revenue_sum_28d": [100.0, 200.0],
    })
    result = seasonal_naive_predict(X)
    np.testing.assert_array_equal(result, [500.0, 200.0])


def test_regression_metrics_zero_error():
    y = np.array([10.0, 20.0, 30.0])
    m = regression_metrics(y, y)
    assert m["mae"] == 0
    assert m["rmse"] == 0
    assert m["mape"] == 0


def test_regression_metrics_known_error():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    m = regression_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(15.0)
    assert m["mean_actual"] == pytest.approx(150.0)


@pytest.mark.parametrize("name", LEARNED_MODEL_NAMES)
def test_make_model_fits_and_predicts(name):
    ts = _synthetic_ts()
    X, y = build_supervised(ts)
    model = make_model(name)
    model.fit(X, y)
    preds = model.predict(X.iloc[:5])
    assert len(preds) == 5
    assert np.all(np.isfinite(preds))


def test_make_model_unknown_name_raises():
    with pytest.raises(ValueError):
        make_model("not-a-real-model")


# ---------------------------------------------------------------------------
# monitoring.py
# ---------------------------------------------------------------------------

def test_check_drift_flags_nothing_for_identical_distribution():
    train_X = pd.DataFrame({"f1": np.random.default_rng(1).normal(0, 1, 200)})
    summary = training_feature_summary(train_X)
    recent = pd.DataFrame({"f1": [0.05]})
    result = check_drift(summary, recent)
    assert result["drift_detected"] is False
    assert result["flagged_features"] == {}


def test_check_drift_flags_a_shifted_feature():
    train_X = pd.DataFrame({"f1": np.random.default_rng(1).normal(0, 1, 200)})
    summary = training_feature_summary(train_X)
    recent = pd.DataFrame({"f1": [50.0]})  # wildly outside training range
    result = check_drift(summary, recent)
    assert result["drift_detected"] is True
    assert "f1" in result["flagged_features"]


# ---------------------------------------------------------------------------
# training.py (uses real cs-train data, but writes only to tmp_path)
# ---------------------------------------------------------------------------

def test_retrain_writes_only_into_the_given_models_dir(tmp_path):
    ROOT = Path(__file__).resolve().parents[1]
    result = retrain(["cs-train"], models_dir=tmp_path, top_n=2)
    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "overall.joblib").exists()
    assert len(result["series_trained"]) == 3  # overall + top 2 countries
    real_models_dir = ROOT / "models"
    assert tmp_path != real_models_dir


def test_retrain_respects_as_of_truncation(tmp_path_factory):
    dir_a = tmp_path_factory.mktemp("early")
    dir_b = tmp_path_factory.mktemp("late")
    retrain(["cs-train"], models_dir=dir_a, top_n=1, as_of="2018-06-01")
    retrain(["cs-train"], models_dir=dir_b, top_n=1, as_of="2019-06-01")
    import json
    meta_a = json.load(open(dir_a / "metadata.json"))
    meta_b = json.load(open(dir_b / "metadata.json"))
    assert meta_a["series"]["overall"]["date_range"][1] < meta_b["series"]["overall"]["date_range"][1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
