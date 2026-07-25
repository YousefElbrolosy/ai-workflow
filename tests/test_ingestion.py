"""Lightweight tests for aavail.ingestion, covering the known messiness in
the AAVAIL invoice exports: inconsistent field names, bad JSON, missing
fields, bad dates, bad-debt adjustment records, and duplicate rows.

Run with: python -m pytest tests/ -q  (or) python tests/test_ingestion.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aavail.ingestion import IngestionError, convert_to_ts, fetch_data, top_countries_by_revenue


def _write(path: Path, records) -> None:
    path.write_text(json.dumps(records))


def test_missing_directory_raises():
    with pytest.raises(IngestionError):
        fetch_data("this/directory/does/not/exist")


def test_empty_directory_raises(tmp_path):
    with pytest.raises(IngestionError):
        fetch_data(tmp_path)


def test_normalizes_field_name_variants(tmp_path):
    _write(
        tmp_path / "invoices-2018-01.json",
        [
            {
                "country": "Germany", "customer_id": 1.0, "invoice": "500001",
                "price": 10.0, "stream_id": "111", "times_viewed": 2,
                "year": "2018", "month": "01", "day": "05",
            }
        ],
    )
    _write(
        tmp_path / "invoices-2018-02.json",
        [
            {
                "country": "Germany", "customer_id": 2.0, "invoice": "500002",
                "total_price": 20.0, "StreamID": "222", "TimesViewed": 4,
                "year": "2018", "month": "02", "day": "05",
            }
        ],
    )
    df = fetch_data(tmp_path)
    assert len(df) == 2
    assert set(df["price"]) == {10.0, 20.0}
    assert set(df["stream_id"]) == {"111", "222"}
    assert set(df["times_viewed"]) == {2, 4}


def test_bad_json_file_is_skipped_not_fatal(tmp_path):
    (tmp_path / "invoices-2018-01.json").write_text("{not valid json")
    _write(
        tmp_path / "invoices-2018-02.json",
        [
            {
                "country": "France", "customer_id": 3.0, "invoice": "500003",
                "price": 5.0, "stream_id": "333", "times_viewed": 1,
                "year": "2018", "month": "02", "day": "10",
            }
        ],
    )
    df, report = fetch_data(tmp_path, return_report=True)
    assert len(df) == 1
    assert "invoices-2018-01.json" in report.files_failed


def test_invalid_date_and_price_are_dropped(tmp_path):
    _write(
        tmp_path / "invoices-2018-01.json",
        [
            {  # bad month
                "country": "Spain", "customer_id": 4.0, "invoice": "500004",
                "price": 1.0, "stream_id": "444", "times_viewed": 1,
                "year": "2018", "month": "13", "day": "01",
            },
            {  # non-numeric price
                "country": "Spain", "customer_id": 5.0, "invoice": "500005",
                "price": "N/A", "stream_id": "555", "times_viewed": 1,
                "year": "2018", "month": "01", "day": "01",
            },
            {  # valid
                "country": "Spain", "customer_id": 6.0, "invoice": "500006",
                "price": 3.0, "stream_id": "666", "times_viewed": 1,
                "year": "2018", "month": "01", "day": "01",
            },
        ],
    )
    df = fetch_data(tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["invoice"] == "500006"


def test_bad_debt_adjustment_records_excluded(tmp_path):
    _write(
        tmp_path / "invoices-2018-01.json",
        [
            {
                "country": "United Kingdom", "customer_id": None, "invoice": "A900001",
                "price": -50000.0, "stream_id": "777", "times_viewed": 0,
                "year": "2018", "month": "01", "day": "15",
            },
            {
                "country": "United Kingdom", "customer_id": 7.0, "invoice": "500007",
                "price": 12.5, "stream_id": "888", "times_viewed": 3,
                "year": "2018", "month": "01", "day": "15",
            },
        ],
    )
    df = fetch_data(tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["invoice_prefix"] != "A"


def test_exact_duplicates_deduplicated(tmp_path):
    record = {
        "country": "Italy", "customer_id": 8.0, "invoice": "500008",
        "price": 9.99, "stream_id": "999", "times_viewed": 2,
        "year": "2018", "month": "01", "day": "20",
    }
    _write(tmp_path / "invoices-2018-01.json", [record, record.copy()])
    df = fetch_data(tmp_path)
    assert len(df) == 1


def test_top_countries_by_revenue_excludes_unspecified(tmp_path):
    _write(
        tmp_path / "invoices-2018-01.json",
        [
            {"country": "Unspecified", "customer_id": 9.0, "invoice": "500009",
             "price": 999.0, "stream_id": "1", "times_viewed": 1,
             "year": "2018", "month": "01", "day": "01"},
            {"country": "France", "customer_id": 10.0, "invoice": "500010",
             "price": 5.0, "stream_id": "2", "times_viewed": 1,
             "year": "2018", "month": "01", "day": "01"},
        ],
    )
    df = fetch_data(tmp_path)
    top = top_countries_by_revenue(df, n=5)
    assert "Unspecified" not in top
    assert "France" in top


def test_convert_to_ts_fills_gap_days(tmp_path):
    _write(
        tmp_path / "invoices-2018-01.json",
        [
            {"country": "Germany", "customer_id": 11.0, "invoice": "500011",
             "price": 10.0, "stream_id": "1", "times_viewed": 1,
             "year": "2018", "month": "01", "day": "01"},
            {"country": "Germany", "customer_id": 12.0, "invoice": "500012",
             "price": 20.0, "stream_id": "2", "times_viewed": 1,
             "year": "2018", "month": "01", "day": "03"},
        ],
    )
    df = fetch_data(tmp_path)
    ts = convert_to_ts(df, country="Germany")
    assert len(ts) == 3
    assert ts.loc["2018-01-02", "revenue"] == 0.0


def test_convert_to_ts_unknown_country_raises(tmp_path):
    _write(
        tmp_path / "invoices-2018-01.json",
        [
            {"country": "Germany", "customer_id": 13.0, "invoice": "500013",
             "price": 10.0, "stream_id": "1", "times_viewed": 1,
             "year": "2018", "month": "01", "day": "01"},
        ],
    )
    df = fetch_data(tmp_path)
    with pytest.raises(IngestionError):
        convert_to_ts(df, country="Nowhere")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
