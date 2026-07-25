"""
Unit tests for aavail/logger.py in isolation from the API.

Every test monkeypatches LOG_DIR to a throwaway tmp_path so nothing here
ever reads or writes the real logs/ directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aavail.logger as logger_module
from aavail.logger import log_event, query_logs


@pytest.fixture(autouse=True)
def isolated_log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(logger_module, "LOG_DIR", tmp_path)
    return tmp_path


def test_log_event_writes_a_json_line_with_a_timestamp(isolated_log_dir):
    record = log_event("predict", {"country": "overall", "predicted_revenue_next_30d": 100.0})
    assert record["kind"] == "predict"
    assert record["country"] == "overall"
    assert "timestamp" in record

    files = list(isolated_log_dir.glob("predict-*.log"))
    assert len(files) == 1


def test_log_event_creates_one_file_per_kind_per_month(isolated_log_dir):
    log_event("train", {"a": 1})
    log_event("predict", {"b": 2})
    assert list(isolated_log_dir.glob("train-*.log"))
    assert list(isolated_log_dir.glob("predict-*.log"))


def test_query_logs_returns_events_newest_first(isolated_log_dir):
    from datetime import datetime, timedelta, timezone
    t0 = datetime.now(timezone.utc)
    log_event("predict", {"seq": 1}, when=t0)
    log_event("predict", {"seq": 2}, when=t0 + timedelta(seconds=1))
    log_event("predict", {"seq": 3}, when=t0 + timedelta(seconds=2))

    results = query_logs(kind="predict")
    assert [r["seq"] for r in results] == [3, 2, 1]


def test_query_logs_filters_by_kind(isolated_log_dir):
    log_event("train", {"which": "train"})
    log_event("predict", {"which": "predict"})
    assert all(r["which"] == "train" for r in query_logs(kind="train"))
    assert all(r["which"] == "predict" for r in query_logs(kind="predict"))


def test_query_logs_respects_limit(isolated_log_dir):
    for i in range(5):
        log_event("predict", {"i": i})
    assert len(query_logs(kind="predict", limit=2)) == 2


def test_query_logs_filters_by_timestamp_range(isolated_log_dir):
    from datetime import datetime, timedelta, timezone
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    log_event("predict", {"day": "jan1"}, when=base)
    log_event("predict", {"day": "jan2"}, when=base + timedelta(days=1))
    log_event("predict", {"day": "jan3"}, when=base + timedelta(days=2))

    results = query_logs(kind="predict", start=base.isoformat(), end=(base + timedelta(days=1)).isoformat())
    days = {r["day"] for r in results}
    assert days == {"jan1", "jan2"}


def test_query_logs_on_empty_directory_returns_empty_list(isolated_log_dir):
    assert query_logs() == []
    assert query_logs(kind="predict") == []


def test_query_logs_skips_malformed_lines(isolated_log_dir):
    log_event("predict", {"ok": True})
    path = next(isolated_log_dir.glob("predict-*.log"))
    with open(path, "a") as fh:
        fh.write("not valid json\n")
    results = query_logs(kind="predict")
    assert len(results) == 1
    assert results[0]["ok"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
