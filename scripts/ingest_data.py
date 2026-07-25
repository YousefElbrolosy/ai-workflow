#!/usr/bin/env python3
"""
Automation entry point for AAVAIL data ingestion.

Reads every monthly invoice export from one or more raw data directories
(the historical "cs-train" export and the newer "cs-production" export are
both, structurally, additional data sources feeding the same pipeline),
cleans and normalizes them with :mod:`aavail.ingestion`, determines the
top-N revenue-generating countries, and writes processed artifacts to disk
for use by EDA notebooks, report generation, and (in part 2) modeling code.

Usage
-----
    python scripts/ingest_data.py
    python scripts/ingest_data.py --data-dir cs-train cs-production --top-n 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aavail.ingestion import IngestionError, convert_to_ts, fetch_data, top_countries_by_revenue

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("ingest_data")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        nargs="+",
        default=["cs-train"],
        help="one or more directories of invoices-*.json files to ingest (default: cs-train)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed",
        help="directory to write processed CSV artifacts to (default: data/processed)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="number of top revenue-generating countries to keep a per-country time series for (default: 10)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    ingestion_reports = {}
    for data_dir in args.data_dir:
        logger.info("ingesting %s", data_dir)
        try:
            df, report = fetch_data(data_dir, return_report=True)
        except IngestionError as exc:
            logger.error("failed to ingest %s: %s", data_dir, exc)
            continue
        logger.info("%s\n%s", data_dir, report.summary())
        df["data_source"] = data_dir
        frames.append(df)
        ingestion_reports[data_dir] = report.as_dict()

    if not frames:
        logger.error("no data sources could be ingested, aborting")
        return 1

    reports_path = out_dir / "ingestion_report.json"
    with open(reports_path, "w") as fh:
        json.dump(ingestion_reports, fh, indent=2)
    logger.info("wrote %s", reports_path)

    transactions = pd.concat(frames, ignore_index=True, sort=False)

    transactions_path = out_dir / "transactions_clean.csv"
    transactions.to_csv(transactions_path, index=False)
    logger.info("wrote %s (%d rows)", transactions_path, len(transactions))

    top_countries = top_countries_by_revenue(transactions, n=args.top_n)
    top_countries_path = out_dir / "top_countries.csv"
    pd.Series(top_countries, name="country").to_csv(top_countries_path, index_label="rank")
    logger.info("top %d countries by revenue: %s", args.top_n, top_countries)

    ts_all = convert_to_ts(transactions)
    ts_all_path = out_dir / "ts_all.csv"
    ts_all.to_csv(ts_all_path)
    logger.info("wrote %s (%d daily rows)", ts_all_path, len(ts_all))

    per_country_frames = []
    for country in top_countries:
        ts_country = convert_to_ts(transactions, country=country)
        ts_country = ts_country.assign(country=country)
        per_country_frames.append(ts_country)
    ts_top_countries = pd.concat(per_country_frames)
    ts_top_countries_path = out_dir / "ts_top_countries.csv"
    ts_top_countries.to_csv(ts_top_countries_path)
    logger.info("wrote %s (%d rows across %d countries)", ts_top_countries_path, len(ts_top_countries), len(top_countries))

    logger.info("ingestion pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
