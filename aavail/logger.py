"""
Structured logging for the AAVAIL revenue API — backs the /logs endpoint.

One append-only JSON-lines file per (event kind, calendar month), e.g.
logs/predict-2019-08.log. JSON lines keep writes atomic-enough for a
single-process Flask app and make the log trivially greppable/parseable
without a database.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


def _ensure_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_event(kind: str, payload: dict, when: Optional[datetime] = None) -> dict:
    """Append one structured event of the given kind (e.g. 'train',
    'predict'). Returns the full record written, including the timestamp."""
    _ensure_dir()
    when = when or datetime.now(timezone.utc)
    record = {"timestamp": when.isoformat(), "kind": kind, **payload}
    path = LOG_DIR / f"{kind}-{when.strftime('%Y-%m')}.log"
    with open(path, "a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    return record


def query_logs(kind: Optional[str] = None, start: Optional[str] = None,
                end: Optional[str] = None, limit: Optional[int] = None) -> list[dict]:
    """Read back logged events, optionally filtered by kind and an
    inclusive [start, end] timestamp range (ISO date/datetime strings),
    newest first, optionally capped at `limit` records."""
    _ensure_dir()
    pattern = f"{kind}-*.log" if kind else "*.log"
    records = []
    for path in sorted(LOG_DIR.glob(pattern)):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if start:
        records = [r for r in records if r["timestamp"] >= start]
    if end:
        records = [r for r in records if r["timestamp"] <= end]

    records.sort(key=lambda r: r["timestamp"], reverse=True)
    if limit:
        records = records[:limit]
    return records
