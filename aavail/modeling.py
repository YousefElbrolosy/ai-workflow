"""
Candidate models for the AAVAIL 30-day revenue forecast, and the metrics
used to compare them. Every model is trained on the same supervised table
produced by aavail.features.build_supervised — same task (predict revenue
summed over the next 30 days), same inputs, so comparisons are apples to
apples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def naive_predict(X: pd.DataFrame) -> np.ndarray:
    """Baseline 1: 'next 30 days = trailing 28 days' (roughly, same as last month)."""
    return X["revenue_sum_28d"].to_numpy()


def seasonal_naive_predict(X: pd.DataFrame) -> np.ndarray:
    """Baseline 2: 'next 30 days = the same 30-day window one year ago',
    falling back to the trailing-28-day baseline where a year of history
    isn't yet available."""
    prior_year = X["revenue_sum_30d_prior_year"].to_numpy().copy()
    has_prior = X["has_prior_year_data"].to_numpy().astype(bool)
    fallback = X["revenue_sum_28d"].to_numpy()
    prior_year[~has_prior] = fallback[~has_prior]
    return prior_year


def make_model(name: str, **hp):
    if name == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(random_state=RANDOM_STATE, **hp))])
    if name == "random_forest":
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **hp)
    if name == "hist_gbm":
        return HistGradientBoostingRegressor(random_state=RANDOM_STATE, **hp)
    if name == "lightgbm":
        return LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1, **hp)
    raise ValueError(f"unknown model name: {name}")


LEARNED_MODEL_NAMES = ("ridge", "random_forest", "hist_gbm", "lightgbm")
ALL_MODEL_NAMES = ("naive", "seasonal_naive") + LEARNED_MODEL_NAMES


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / np.clip(np.abs(y_true), 1e-6, None)) * 100)
    return {"mae": mae, "rmse": rmse, "mape": mape, "mean_actual": float(y_true.mean())}
