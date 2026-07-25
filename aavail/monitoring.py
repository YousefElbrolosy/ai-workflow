"""
Lightweight feature-drift monitoring for the AAVAIL revenue API.

Compares the distribution of engineered features at prediction time against
the distribution the currently-deployed model was trained on. This is
intentionally simple (z-scores on means, no external dependencies) — the
point is to have *something* automated flagging "the world has moved since
this model was trained" for the handful of active users this service has,
not a full statistical monitoring platform.
"""

from __future__ import annotations

import pandas as pd

Z_THRESHOLD_DEFAULT = 3.0


def training_feature_summary(X: pd.DataFrame) -> dict:
    """Snapshot of each feature's mean/std at training time. Saved alongside
    the model so drift can be checked without keeping the full training set
    in memory at serving time."""
    return {
        "n_rows": int(len(X)),
        "mean": X.mean().to_dict(),
        "std": X.std(ddof=0).replace(0, 1e-9).to_dict(),
    }


def check_drift(train_summary: dict, recent_X: pd.DataFrame, z_threshold: float = Z_THRESHOLD_DEFAULT) -> dict:
    """Compare `recent_X` (one or more recent feature rows) against the
    training-time summary. Returns per-feature z-scores and the list of
    features whose recent mean has drifted more than `z_threshold` training
    standard deviations from the training mean."""
    train_mean = pd.Series(train_summary["mean"])
    train_std = pd.Series(train_summary["std"])
    recent_mean = recent_X.reindex(columns=train_mean.index).mean()

    z = ((recent_mean - train_mean) / train_std).fillna(0.0)
    flagged = z[z.abs() > z_threshold].sort_values(key=abs, ascending=False)

    return {
        "z_threshold": z_threshold,
        "n_recent_rows": int(len(recent_X)),
        "n_features_checked": int(len(train_mean)),
        "flagged_features": {k: round(float(v), 2) for k, v in flagged.items()},
        "max_abs_z": round(float(z.abs().max()) if len(z) else 0.0, 2),
        "drift_detected": bool(len(flagged) > 0),
    }
