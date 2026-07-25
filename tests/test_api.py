"""
TDD-style tests for the AAVAIL revenue API (api/app.py).

Covers the contract each endpoint must hold, the edge cases the case study
specifically calls out (arbitrary as-of dates, not just "the latest date"),
and a couple of checks that the API anticipates scale/load (in-memory
caching, not re-ingesting on every request) and drift (the /drift endpoint
and predict's embedded drift check both function).

Run with: python -m pytest tests/test_api.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.app import app as flask_app
import api.app as api_module


@pytest.fixture(scope="module")
def client():
    flask_app.config.update(TESTING=True)
    return flask_app.test_client()


@pytest.fixture(scope="module", autouse=True)
def trained_model(client):
    """Train once, on cs-train, before any test in this module runs."""
    resp = client.post("/train", json={"data_dirs": ["cs-train"]})
    assert resp.status_code == 200, resp.get_json()
    return resp.get_json()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_train_reports_overall_and_top_countries(trained_model):
    trained = trained_model["result"]["series_trained"]
    assert "overall" in trained
    assert len(trained) == 11  # overall + 10 countries
    assert "United Kingdom" in trained


def test_predict_end_of_month(client):
    resp = client.post("/predict", json={"country": "overall", "date": "2019-07-31"})
    body = resp.get_json()
    assert resp.status_code == 200, body
    assert body["country"] == "overall"
    assert body["horizon_days"] == 30
    assert isinstance(body["predicted_revenue_next_30d"], (int, float))
    assert body["predicted_revenue_next_30d"] > 0


def test_predict_mid_month_matches_managers_who_project_on_the_15th(client):
    """Management explicitly needs projections on dates other than the last
    day of the data, e.g. the 15th — not just 'the latest date'."""
    resp = client.post("/predict", json={"country": "overall", "date": "2019-01-15"})
    assert resp.status_code == 200, resp.get_json()


def test_predict_country_is_case_insensitive(client):
    resp = client.post("/predict", json={"country": "united kingdom", "date": "2019-07-31"})
    assert resp.status_code == 200, resp.get_json()
    assert resp.get_json()["country"] == "United Kingdom"


def test_predict_missing_params_returns_400(client):
    resp = client.post("/predict", json={"country": "overall"})
    assert resp.status_code == 400
    resp = client.post("/predict", json={"date": "2019-07-31"})
    assert resp.status_code == 400


def test_predict_unknown_country_returns_400_with_valid_options(client):
    resp = client.post("/predict", json={"country": "Narnia", "date": "2019-07-31"})
    assert resp.status_code == 400
    assert "Valid options" in resp.get_json()["error"]


def test_predict_date_before_available_history_returns_400(client):
    resp = client.post("/predict", json={"country": "overall", "date": "2017-12-01"})
    assert resp.status_code == 400
    assert "insufficient" in resp.get_json()["error"].lower()


def test_predict_date_outside_ingested_range_returns_400(client):
    resp = client.post("/predict", json={"country": "overall", "date": "2030-01-01"})
    assert resp.status_code == 400


def test_predict_without_a_trained_model_returns_503(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api_module, "MODELS_DIR", tmp_path)
    api_module._state["metadata"] = None
    try:
        resp = client.post("/predict", json={"country": "overall", "date": "2019-07-31"})
        assert resp.status_code == 503
    finally:
        api_module._state["metadata"] = None  # force a clean reload from the real MODELS_DIR


def test_drift_endpoint_returns_a_verdict(client):
    resp = client.get("/drift", query_string={"country": "overall", "date": "2019-07-31"})
    body = resp.get_json()
    assert resp.status_code == 200, body
    assert "drift_detected" in body
    assert "max_abs_z" in body


def test_drift_endpoint_requires_params(client):
    resp = client.get("/drift", query_string={"country": "overall"})
    assert resp.status_code == 400


def test_logs_endpoint_reflects_predict_calls(client):
    client.post("/predict", json={"country": "overall", "date": "2019-06-30"})
    resp = client.get("/logs", query_string={"type": "predict", "limit": 5})
    assert resp.status_code == 200
    body = resp.get_json()
    assert len(body) >= 1
    assert all(r["kind"] == "predict" for r in body)


def test_logs_endpoint_reflects_train_calls(client):
    resp = client.get("/logs", query_string={"type": "train", "limit": 5})
    body = resp.get_json()
    assert resp.status_code == 200
    assert any(r["kind"] == "train" for r in body)


def test_predict_reuses_cached_transactions_across_calls(client):
    """Anticipates load: ingesting 21 JSON files on every single request
    would not scale even to a handful of concurrent users querying
    different countries — the transactions cache must be shared."""
    client.post("/predict", json={"country": "overall", "date": "2019-07-31"})
    cached_after_first = api_module._state["transactions"]
    assert cached_after_first is not None
    client.post("/predict", json={"country": "Germany", "date": "2019-07-31"})
    assert api_module._state["transactions"] is cached_after_first


def test_train_invalidates_caches(client):
    """A stale in-memory model must never be served after a retrain."""
    client.post("/predict", json={"country": "overall", "date": "2019-07-31"})
    assert api_module._state["models"]  # something is cached

    resp = client.post("/train", json={"data_dirs": ["cs-train"]})
    assert resp.status_code == 200
    assert api_module._state["models"] == {}
    assert api_module._state["transactions"] is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
