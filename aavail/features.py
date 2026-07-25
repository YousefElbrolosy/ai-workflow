"""
Feature engineering for the AAVAIL 30-day revenue forecast.

Standardizes every model on the same task: given all data up to and
including day t, predict total revenue over the next 30 days (t+1 .. t+30).
Works on the daily time series produced by aavail.ingestion.convert_to_ts,
for the overall series or any single country.
"""

from __future__ import annotations

import pandas as pd

HORIZON = 30
LAG_WINDOWS = (7, 14, 28, 70)


def _build_features(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Trailing-window features computable as of each date in df — no
    knowledge of the future is required, so this is safe to call both for
    training (before slicing to rows with a valid target) and at inference
    time on the most recent date available."""
    feats = pd.DataFrame(index=df.index)

    for col in ["revenue", "purchases", "unique_streams", "unique_customers", "total_views"]:
        for w in LAG_WINDOWS:
            trailing = df[col].rolling(w, min_periods=w)
            feats[f"{col}_sum_{w}d"] = trailing.sum()
            feats[f"{col}_mean_{w}d"] = trailing.mean()
        feats[f"{col}_lag_{horizon}d"] = df[col].shift(horizon)
        mean28 = df[col].rolling(28, min_periods=28).mean()
        feats[f"{col}_trend_28d"] = mean28 - mean28.shift(28)

    # Same 30-day window one year earlier, when available — captures annual
    # seasonality (e.g. the holiday build-up / year-end gap found in Part 1).
    # Optional: most series don't have a full year of prior history, so a
    # missing value is filled (not dropped) and flagged with its own column
    # rather than costing the row entirely.
    prior_year = df["revenue"].rolling(30, min_periods=30).sum().shift(365)
    feats["has_prior_year_data"] = prior_year.notna().astype(int)
    feats["revenue_sum_30d_prior_year"] = prior_year.fillna(0.0)

    feats["dow"] = df.index.dayofweek
    feats["day_of_month"] = df.index.day
    feats["month"] = df.index.month
    feats["is_year_end_window"] = ((df.index.month == 12) & (df.index.day >= 21)).astype(int)

    return feats


def build_supervised(ts: pd.DataFrame, horizon: int = HORIZON) -> tuple[pd.DataFrame, pd.Series]:
    """Turn a daily time series (columns: revenue, purchases, unique_streams,
    unique_customers, total_views; DatetimeIndex) into a supervised learning
    table.

    Returns (X, y) aligned on the same index (the "as-of" date t). Rows near
    the start (not enough trailing history) and near the end (not enough
    future days to sum a full 30-day target) are dropped.
    """
    df = ts.copy().asfreq("D").fillna(0.0)
    feats = _build_features(df, horizon=horizon)

    target = df["revenue"].shift(-1).rolling(horizon, min_periods=horizon).sum().shift(-(horizon - 1))
    # shift(-1) + rolling(horizon).sum() + shift(-(horizon-1)) == sum of t+1..t+horizon at row t
    target.name = "target_revenue_next_30d"

    valid = feats.notna().all(axis=1) & target.notna()
    return feats.loc[valid], target.loc[valid]


def feature_row_for_date(ts: pd.DataFrame, as_of_date, horizon: int = HORIZON) -> pd.DataFrame:
    """The single feature row usable to forecast the next `horizon` days as
    of `as_of_date` — the date a manager is standing at when they ask for a
    projection (the 15th, the end of the month, ...), not necessarily the
    most recent date in the data.

    `ts` is truncated to `<= as_of_date` *before* any rolling/shift feature
    is computed, so nothing past that date can leak into the result. Raises
    ValueError if `as_of_date` falls outside the range covered by `ts`.
    """
    df = ts.copy().asfreq("D").fillna(0.0)
    as_of_date = pd.Timestamp(as_of_date)
    if as_of_date < df.index.min() or as_of_date > df.index.max():
        raise ValueError(
            f"as_of_date {as_of_date.date()} is outside the ingested data range "
            f"({df.index.min().date()} .. {df.index.max().date()})"
        )
    df = df.loc[:as_of_date]
    feats = _build_features(df, horizon=horizon)
    return feats.loc[[as_of_date]]


def latest_feature_row(ts: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """The single feature row usable to forecast the next `horizon` days
    from "today" (the last date present in `ts`), for deployment-time
    inference — no target/future data required."""
    df = ts.copy().asfreq("D").fillna(0.0)
    feats = _build_features(df, horizon=horizon)
    return feats.iloc[[-1]]
