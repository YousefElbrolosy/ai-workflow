"""
Data ingestion library for the AAVAIL revenue-prediction service.

The raw data arrive as one JSON file per month (e.g. ``invoices-2018-01.json``),
each holding a list of transaction-level records. The files come from more than
one export process, so field names are not uniform across files
(``price``/``total_price``, ``stream_id``/``StreamID``, ``times_viewed``/``TimesViewed``).

This module is responsible for:
    * discovering and loading every JSON file in a data directory,
    * normalizing schema differences between files,
    * validating and coercing types, catching the input errors that are known
      to occur in this data source,
    * returning a single, clean, transaction-level feature matrix (a
      ``pandas.DataFrame``) that downstream EDA and modeling code can rely on.

Typical usage
-------------
>>> from aavail.ingestion import fetch_data, convert_to_ts, top_countries_by_revenue
>>> df = fetch_data("cs-train")
>>> top10 = top_countries_by_revenue(df, n=10)
>>> ts_all = convert_to_ts(df)
>>> ts_uk = convert_to_ts(df, country="United Kingdom")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Schema handling
# ----------------------------------------------------------------------------

# Different export batches used different capitalization/naming for the same
# field. Map every variant seen in the data onto one canonical name.
COLUMN_ALIASES = {
    "streamid": "stream_id",
    "stream_id": "stream_id",
    "timesviewed": "times_viewed",
    "times_viewed": "times_viewed",
    "total_price": "price",
    "price": "price",
    "country": "country",
    "customer_id": "customer_id",
    "invoice": "invoice",
    "year": "year",
    "month": "month",
    "day": "day",
}

REQUIRED_CANONICAL_COLUMNS = {
    "country",
    "customer_id",
    "invoice",
    "price",
    "stream_id",
    "times_viewed",
    "year",
    "month",
    "day",
}

# Country labels present in the source data that do not represent a real
# purchasing market and should be excluded when ranking countries by revenue.
NON_COUNTRY_LABELS = {"unspecified", "european community"}


class IngestionError(RuntimeError):
    """Raised for unrecoverable problems with a data source (bad directory,
    zero readable files, etc.). Per-record and per-file problems are instead
    logged and repaired/skipped so that one malformed file cannot take down
    the whole pipeline."""


@dataclass
class IngestionReport:
    """Bookkeeping for what happened during ingestion, so cleaning decisions
    are auditable rather than silent."""

    files_found: int = 0
    files_loaded: int = 0
    files_failed: list = field(default_factory=list)
    rows_read: int = 0
    rows_dropped_bad_json_record: int = 0
    rows_dropped_missing_required_field: int = 0
    rows_dropped_invalid_date: int = 0
    rows_dropped_invalid_price: int = 0
    rows_dropped_bad_debt_adjustment: int = 0
    rows_dropped_exact_duplicate: int = 0
    rows_kept: int = 0

    def as_dict(self) -> dict:
        return {
            "files_found": self.files_found,
            "files_loaded": self.files_loaded,
            "files_failed": list(self.files_failed),
            "rows_read": self.rows_read,
            "rows_dropped_bad_json_record": self.rows_dropped_bad_json_record,
            "rows_dropped_missing_required_field": self.rows_dropped_missing_required_field,
            "rows_dropped_invalid_date": self.rows_dropped_invalid_date,
            "rows_dropped_invalid_price": self.rows_dropped_invalid_price,
            "rows_dropped_bad_debt_adjustment": self.rows_dropped_bad_debt_adjustment,
            "rows_dropped_exact_duplicate": self.rows_dropped_exact_duplicate,
            "rows_kept": self.rows_kept,
        }

    def summary(self) -> str:
        lines = [
            f"files found / loaded : {self.files_found} / {self.files_loaded}",
        ]
        if self.files_failed:
            lines.append(f"files failed to load  : {self.files_failed}")
        lines += [
            f"rows read             : {self.rows_read}",
            f"  - dropped, malformed record        : {self.rows_dropped_bad_json_record}",
            f"  - dropped, missing required field   : {self.rows_dropped_missing_required_field}",
            f"  - dropped, invalid date              : {self.rows_dropped_invalid_date}",
            f"  - dropped, invalid/non-numeric price : {self.rows_dropped_invalid_price}",
            f"  - dropped, bad-debt adjustment (A-*) : {self.rows_dropped_bad_debt_adjustment}",
            f"  - dropped, exact duplicate row       : {self.rows_dropped_exact_duplicate}",
            f"rows kept             : {self.rows_kept}",
        ]
        return "\n".join(lines)


def _normalize_columns(record_keys: Iterable[str]) -> dict:
    """Build a rename map from the raw keys present in one record to the
    canonical column names used everywhere downstream."""
    rename = {}
    for key in record_keys:
        canonical = COLUMN_ALIASES.get(key.lower())
        if canonical is not None:
            rename[key] = canonical
    return rename


def _load_one_file(path: Path, report: IngestionReport) -> Optional[pd.DataFrame]:
    """Load and schema-normalize a single JSON file. Returns None (and logs)
    if the file cannot be read at all."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        logger.warning("skipping unreadable file %s (%s)", path.name, exc)
        report.files_failed.append(path.name)
        return None

    if not isinstance(raw, list) or len(raw) == 0:
        logger.warning("skipping %s: expected a non-empty JSON array of records", path.name)
        report.files_failed.append(path.name)
        return None

    report.rows_read += len(raw)

    # Some records could in principle be malformed (not a dict). Filter those
    # out defensively rather than letting pd.DataFrame choke on them.
    clean_records = [r for r in raw if isinstance(r, dict)]
    n_bad = len(raw) - len(clean_records)
    if n_bad:
        report.rows_dropped_bad_json_record += n_bad

    if not clean_records:
        return None

    df = pd.DataFrame.from_records(clean_records)
    rename_map = _normalize_columns(df.columns)
    df = df.rename(columns=rename_map)

    missing = REQUIRED_CANONICAL_COLUMNS - set(df.columns)
    if missing:
        # File is missing a required field under every alias we know about.
        # Rather than aborting the whole run, drop the file and keep going -
        # a single malformed export should not block ingestion of the rest.
        logger.warning("skipping %s: missing required field(s) %s", path.name, sorted(missing))
        report.files_failed.append(path.name)
        return None

    df["source_file"] = path.name
    return df[list(REQUIRED_CANONICAL_COLUMNS) + ["source_file"]]


def fetch_data(
    data_dir: Union[str, Path],
    file_pattern: str = "invoices-*.json",
    return_report: bool = False,
):
    """Read every invoice JSON file in ``data_dir`` and return one clean,
    transaction-level feature matrix.

    Handles, on purpose, the known problems in this data source:
      * inconsistent field naming across export batches
      * missing/invalid dates
      * non-numeric or missing prices
      * a handful of extreme negative "bad debt adjustment" records
        (invoice numbers prefixed with 'A') that are accounting write-offs,
        not customer purchases, and would otherwise distort daily revenue
      * exact duplicate rows introduced by the export process

    Parameters
    ----------
    data_dir : str or Path
        Directory containing ``invoices-*.json`` files (e.g. ``cs-train``).
    file_pattern : str
        Glob pattern used to discover files inside ``data_dir``.
    return_report : bool
        If True, return ``(df, report)`` where ``report`` is an
        :class:`IngestionReport` describing what was read/dropped and why.
        The report is always attached to the returned frame's
        ``.attrs['ingestion_report']`` regardless of this flag.

    Returns
    -------
    pandas.DataFrame (or (DataFrame, IngestionReport) if return_report=True)
        One row per transaction line with columns:
        ``date, year, month, day, country, customer_id, invoice,
        invoice_prefix, invoice_clean, stream_id, price, times_viewed,
        source_file``.

    Raises
    ------
    IngestionError
        If ``data_dir`` does not exist, is not a directory, contains no
        matching files, or none of the matching files could be parsed into
        usable records.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise IngestionError(f"data directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise IngestionError(f"not a directory: {data_dir}")

    report = IngestionReport()
    paths = sorted(data_dir.glob(file_pattern))
    report.files_found = len(paths)
    if not paths:
        raise IngestionError(
            f"no files matching '{file_pattern}' found in {data_dir}"
        )

    frames = []
    for path in paths:
        frame = _load_one_file(path, report)
        if frame is not None:
            frames.append(frame)
            report.files_loaded += 1

    if not frames:
        raise IngestionError(
            f"found {len(paths)} file(s) in {data_dir} but none could be parsed"
        )

    df = pd.concat(frames, ignore_index=True, sort=False)

    # ---- type coercion -----------------------------------------------
    df["country"] = df["country"].astype(str).str.strip()
    df["invoice"] = df["invoice"].astype(str).str.strip()
    df["stream_id"] = df["stream_id"].astype(str).str.strip()

    # invoice ids carry an incidental letter prefix (e.g. 'C512770',
    # 'A506401'). Split it out so invoices can be matched/counted on their
    # numeric id alone, while keeping the prefix as its own signal.
    extracted = df["invoice"].str.extract(r"^(?P<invoice_prefix>[A-Za-z]*)(?P<invoice_clean>.*)$")
    df["invoice_prefix"] = extracted["invoice_prefix"]
    df["invoice_clean"] = extracted["invoice_clean"]

    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce").astype("Int64")

    price_before = df["price"].copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    n_bad_price = int((price_before.notna() & df["price"].isna()).sum() + price_before.isna().sum())
    if n_bad_price:
        report.rows_dropped_invalid_price += n_bad_price
    df = df[df["price"].notna()]

    df["times_viewed"] = pd.to_numeric(df["times_viewed"], errors="coerce").fillna(0).astype(int)

    # ---- date construction ---------------------------------------------
    date = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=df["day"]),
        errors="coerce",
    )
    n_bad_date = int(date.isna().sum())
    if n_bad_date:
        report.rows_dropped_invalid_date += n_bad_date
    df["date"] = date
    df = df[df["date"].notna()]
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    # ---- required-field completeness ------------------------------------
    before = len(df)
    df = df.dropna(subset=["country", "invoice", "stream_id", "price", "date"])
    df = df[(df["country"] != "") & (df["country"].str.lower() != "nan")]
    report.rows_dropped_missing_required_field += before - len(df)

    # ---- known data-quality issue: bad-debt write-offs -------------------
    # A handful of records (invoice prefix 'A') are extreme-magnitude,
    # negative-price accounting adjustments for written-off debt, not actual
    # customer purchases. They are not part of the revenue-generating
    # process this model targets, so they are excluded and the exclusion is
    # logged for transparency.
    is_bad_debt = df["invoice_prefix"].str.upper().eq("A")
    report.rows_dropped_bad_debt_adjustment += int(is_bad_debt.sum())
    df = df[~is_bad_debt]

    # ---- exact duplicate rows --------------------------------------------
    before = len(df)
    dedup_cols = ["country", "customer_id", "invoice", "stream_id", "price", "times_viewed", "date"]
    df = df.drop_duplicates(subset=dedup_cols)
    report.rows_dropped_exact_duplicate += before - len(df)

    df = df.sort_values("date").reset_index(drop=True)
    report.rows_kept = len(df)

    ordered_cols = [
        "date", "year", "month", "day",
        "country", "customer_id",
        "invoice", "invoice_prefix", "invoice_clean",
        "stream_id", "price", "times_viewed",
        "source_file",
    ]
    df = df[ordered_cols]
    df.attrs["ingestion_report"] = report

    logger.info("ingestion complete:\n%s", report.summary())

    if return_report:
        return df, report
    return df


def top_countries_by_revenue(df: pd.DataFrame, n: int = 10, exclude_uk: bool = False) -> list:
    """Return the ``n`` country names with the highest total revenue in
    ``df``, excluding non-country labels such as 'Unspecified'.

    Parameters
    ----------
    exclude_uk : bool
        AAVAIL's home market (United Kingdom) dominates raw transaction
        volume; set True to rank the remaining international markets only.
    """
    mask = ~df["country"].str.lower().isin(NON_COUNTRY_LABELS)
    if exclude_uk:
        mask &= df["country"] != "United Kingdom"
    revenue_by_country = (
        df[mask].groupby("country")["price"].sum().sort_values(ascending=False)
    )
    return revenue_by_country.head(n).index.tolist()


def convert_to_ts(df: pd.DataFrame, country: Optional[str] = None) -> pd.DataFrame:
    """Aggregate a transaction-level feature matrix into a daily time series
    suitable for forecasting.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`fetch_data`.
    country : str, optional
        If given, restrict to that country before aggregating. If None,
        aggregate across all countries.

    Returns
    -------
    pandas.DataFrame indexed by ``date`` (one row per calendar day spanning
    the observed range, gaps filled with zero-activity days) with columns:
    ``revenue, purchases, unique_streams, unique_customers, total_views``.
    """
    working = df if country is None else df[df["country"] == country]
    if country is not None and working.empty:
        raise IngestionError(f"no records found for country={country!r}")

    daily = (
        working.groupby("date")
        .agg(
            revenue=("price", "sum"),
            purchases=("invoice_clean", "nunique"),
            unique_streams=("stream_id", "nunique"),
            unique_customers=("customer_id", "nunique"),
            total_views=("times_viewed", "sum"),
        )
        .sort_index()
    )

    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range)
    daily.index.name = "date"
    fill_zero_cols = ["revenue", "purchases", "unique_streams", "unique_customers", "total_views"]
    daily[fill_zero_cols] = daily[fill_zero_cols].fillna(0)
    daily["revenue"] = daily["revenue"].astype(float).round(2)
    for col in ["purchases", "unique_streams", "unique_customers", "total_views"]:
        daily[col] = daily[col].astype(int)

    return daily
