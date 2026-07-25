#!/usr/bin/env python3
"""
Simulate real usage of the deployed (Dockerized) AAVAIL revenue API.

Walks forward day by day over a date range, calling POST /predict for each
target series as if that day were "today" — the model only ever sees data
up to the simulated day, via /train's `as_of` truncation, so this never
lets the future leak into a prediction. Periodically re-trains (like a
nightly/weekly retrain job) to mimic how the service would actually run in
production. Every prediction is captured by the API's own /predict log, so
after this script finishes, scripts/analyze_performance.py can compare
logged predictions against revenue that has since actually been observed.

Usage
-----
    # with the API running in Docker (see README section in Part 3 report):
    #   docker run --rm -p 8080:8080 \
    #     -v $(pwd)/cs-production:/app/cs-production aavail-revenue-api
    python scripts/simulate_queries.py \
        --base-url http://localhost:8080 \
        --start 2019-08-01 --end 2019-12-31 --retrain-every 7
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timedelta

import requests

DEFAULT_COUNTRIES = [
    "overall", "United Kingdom", "EIRE", "Germany", "France", "Norway",
    "Spain", "Hong Kong", "Portugal", "Singapore", "Netherlands",
]


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8080")
    p.add_argument("--start", required=True, help="YYYY-MM-DD, first simulated day")
    p.add_argument("--end", required=True, help="YYYY-MM-DD, last simulated day")
    p.add_argument("--retrain-every", type=int, default=7, help="days between retrains (0 = never retrain during the run)")
    p.add_argument("--data-dirs", nargs="+", default=["cs-train", "cs-production"])
    p.add_argument("--countries", nargs="+", default=DEFAULT_COUNTRIES)
    p.add_argument("--sleep", type=float, default=0.0, help="seconds to sleep between predict calls (0 = as fast as possible)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    session = requests.Session()

    def train(as_of: date):
        t0 = time.time()
        resp = session.post(f"{args.base_url}/train",
                             json={"data_dirs": args.data_dirs, "as_of": as_of.isoformat()})
        resp.raise_for_status()
        print(f"[train] as_of={as_of} -> {resp.json()['result']['series_trained']} ({time.time()-t0:.1f}s)")

    def predict(country: str, day: date) -> bool:
        resp = session.post(f"{args.base_url}/predict", json={"country": country, "date": day.isoformat()})
        if resp.status_code != 200:
            print(f"[predict] {country} {day} FAILED: {resp.status_code} {resp.json().get('error')}", file=sys.stderr)
            return False
        return True

    resp = session.get(f"{args.base_url}/health", timeout=5)
    resp.raise_for_status()
    print(f"API reachable at {args.base_url}")

    # a model must exist before day 1's predictions
    train(as_of=start - timedelta(days=1))

    n_predicted, n_failed, n_days = 0, 0, 0
    for day in daterange(start, end):
        if args.retrain_every and n_days > 0 and n_days % args.retrain_every == 0:
            train(as_of=day)
        for country in args.countries:
            ok = predict(country, day)
            n_predicted += int(ok)
            n_failed += int(not ok)
            if args.sleep:
                time.sleep(args.sleep)
        n_days += 1
        if n_days % 10 == 0:
            print(f"...simulated {n_days} days ({day}), {n_predicted} predictions logged, {n_failed} failed")

    print(f"done: {n_days} days simulated, {n_predicted} predictions logged, {n_failed} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
