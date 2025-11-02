
"""
rolling_origin_cv.py
--------------------
A production-ready Rolling-Origin (expanding window) cross-validation splitter
that respects *global calendar time* regardless of row order or grouping.

Why this exists
---------------
- Scikit-learn's TimeSeriesSplit splits by row index, which can be wrong when
  your data are sorted by ['group', 'datetime'] (common in panel datasets).
- This splitter *always* splits by actual dates taken from a column you specify
  (e.g., 'datetime'), so every fold is a clean past->future evaluation across
  all groups (e.g., provinces).

Key features
------------
- Expanding-window training (a.k.a rolling-origin) with a configurable GAP to
  avoid label leakage when you compute lag/rolling features near the boundary.
- Horizon can span multiple days (e.g., 1-day or 30-day blocks).
- Works with pandas DataFrame `X` directly (set `date_col="datetime"`), or
  pass the datetimes via the `groups` argument to `.split()`.
- Sklearn-compatible: use with cross_val_score, custom CV loops, or HPO (Optuna).

Basic usage
-----------
>>> from rolling_origin_cv import RollingOriginTimeSeriesSplit
>>> cv = RollingOriginTimeSeriesSplit(
...     date_col="datetime",       # column in X
...     min_train_days=365*3,      # size of initial training window (days)
...     step_days=30,              # how far the origin moves each fold (days)
...     horizon_days=7,            # test window length (days)
...     gap_days=30,               # purge window between train end and test start
...     max_splits=None            # limit number of folds if desired
... )
>>> for i, (tr, te) in enumerate(cv.split(X)):
...     print(i, X.loc[tr, "datetime"].min(), X.loc[tr, "datetime"].max(),
...              X.loc[te, "datetime"].min(), X.loc[te, "datetime"].max())

Integration with cross_val_score
--------------------------------
>>> from sklearn.metrics import make_scorer, mean_absolute_error
>>> from sklearn.model_selection import cross_val_score
>>> scorer = make_scorer(mean_absolute_error, greater_is_better=False)
>>> scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
>>> scores.mean()

Notes
-----
- Dates are normalized to midnight (date-level). If you have sub-daily timestamps,
  they are bucketed by calendar day for split logic.
- For 1-step-ahead forecasting with lag features up to L days, set gap_days >= L.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple, List, Dict, Any
import numpy as np
import pandas as pd

@dataclass
class RollingOriginTimeSeriesSplit:
    date_col: Optional[str] = None
    min_train_days: int = 365
    step_days: int = 30
    horizon_days: int = 7
    gap_days: int = 0
    max_splits: Optional[int] = None

    def __post_init__(self):
        for name in ["min_train_days", "step_days", "horizon_days", "gap_days"]:
            val = getattr(self, name)
            if not isinstance(val, int) or val < 0:
                raise ValueError(f"{name} must be a non-negative int, got {val!r}")
        if self.horizon_days == 0:
            raise ValueError("horizon_days must be >= 1")
        if self.date_col is None:
            # Allowed only if user will pass datetime array via groups in split()
            pass

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return len(self._compute_windows(X, groups))

    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        windows = self._compute_windows(X, groups)
        # Pre-extract date series to create masks quickly
        dates = self._extract_dates(X, groups)
        # Normalize to date
        # dates = pd.to_datetime(dates).dt.normalize().values.astype('datetime64[D]')
        dates = pd.to_datetime(dates).dt.normalize().values.astype('datetime64[D]')

        for (train_end, test_start, test_end) in windows:
            train_mask = dates <= train_end
            test_mask  = (dates >= test_start) & (dates <= test_end)
            tr_idx = np.flatnonzero(train_mask)
            te_idx = np.flatnonzero(test_mask)
            if tr_idx.size == 0 or te_idx.size == 0:
                # Skip ill-formed windows silently
                continue
            yield tr_idx, te_idx

    # ---------- Helpers ----------
    def _extract_dates(self, X, groups=None) -> pd.Series:
        if self.date_col is not None:
            # X can be a pandas DataFrame or any structure with column access
            if hasattr(X, "__getitem__"):
                dates = pd.to_datetime(pd.Series(X[self.date_col])).copy()
            else:
                raise TypeError("X must support column access for date_col.")
        else:
            if groups is None:
                raise ValueError("Either provide date_col in __init__ or pass datetime array via groups.")
            dates = pd.to_datetime(pd.Series(groups)).copy()
        if dates.isna().any():
            raise ValueError("Found NA in dates. Please clean your datetime column.")
        return dates

    def _compute_windows(self, X, groups=None) -> List[Tuple[np.datetime64, np.datetime64, np.datetime64]]:
        dates = self._extract_dates(X, groups)
        # normalize to date-level
        d_unique = pd.to_datetime(dates).dt.normalize().drop_duplicates().sort_values().to_numpy(dtype='datetime64[D]')
        if d_unique.size == 0:
            return []
        first = d_unique[0]
        last  = d_unique[-1]

        # initial train end
        train_end = first + np.timedelta64(self.min_train_days - 1, 'D')
        windows: List[Tuple[np.datetime64, np.datetime64, np.datetime64]] = []
        n = 0
        while True:
            test_start = train_end + np.timedelta64(self.gap_days + 1, 'D')
            test_end   = test_start + np.timedelta64(self.horizon_days - 1, 'D')
            # stop if beyond available range
            if test_end > last:
                break
            windows.append((train_end, test_start, test_end))
            n += 1
            if self.max_splits is not None and n >= self.max_splits:
                break
            # advance the origin
            train_end = train_end + np.timedelta64(self.step_days, 'D')
        return windows

    # Convenience: human-readable table of folds
    def describe_folds(self, X, groups=None) -> pd.DataFrame:
        wins = self._compute_windows(X, groups)
        if not wins:
            return pd.DataFrame(columns=["fold", "train_end", "test_start", "test_end"])
        rows = [
            {"fold": i, "train_end": str(te), "test_start": str(ts), "test_end": str(te2)}
            for i, (te, ts, te2) in enumerate(wins, 1)
        ]
        return pd.DataFrame(rows)


# ---- Utility for quick evaluation loops ----
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_cv(estimator, X: pd.DataFrame, y, cv: RollingOriginTimeSeriesSplit,
                sample_weight=None) -> pd.DataFrame:
    """
    Run a manual CV loop over rolling-origin folds and return per-fold metrics.
    """
    results = []
    for i, (tr, te) in enumerate(cv.split(X, y)):
        est = estimator
        # Clone if available (sklearn estimators); otherwise fit in-place.
        try:
            from sklearn.base import clone
            est = clone(estimator)
        except Exception:
            pass
        est.fit(X.iloc[tr], y.iloc[tr], **({"sample_weight": sample_weight.iloc[tr]} if sample_weight is not None else {}))
        y_pred = est.predict(X.iloc[te])
        mae  = mean_absolute_error(y.iloc[te], y_pred)
        rmse = mean_squared_error(y.iloc[te], y_pred, squared=False)
        r2   = r2_score(y.iloc[te], y_pred)
        results.append({"fold": i+1, "MAE": mae, "RMSE": rmse, "R2": r2,
                        "n_train": int(len(tr)), "n_test": int(len(te))})
    return pd.DataFrame(results)
