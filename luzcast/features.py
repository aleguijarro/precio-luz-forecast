"""Feature engineering on a day x hour price matrix.

Working with a matrix (one row per local day, 24 columns for local hours) makes the
"direct multi-horizon" setup explicit: to forecast day D with a k-day-ahead model, every
feature is computed from days <= D-k, i.e. only from information available when the
forecast is issued. This avoids look-ahead leakage by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar import is_holiday

TZ = "Europe/Madrid"
HOURS = list(range(24))

FEATURES = [
    "hour",
    "dow",
    "month",
    "is_offday",
    "doy_sin",
    "doy_cos",
    "lag_h",
    "lag_h_prev",
    "lag_h_next",
    "lag_mean",
    "lag_min",
    "lag_max",
    "lag_std",
    "lag_offday",
    "week_h",
    "week_mean",
    "trend_mean",
]


def to_daily_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Pivot hourly UTC prices into a (local day x local hour) matrix with a complete day index.

    DST days have 23 or 25 local hours: duplicates are averaged and the missing hour is
    filled from its neighbours, which keeps every row at exactly 24 values.
    """
    if prices.empty:
        return pd.DataFrame(columns=HOURS, dtype=float)
    local = prices["datetime"].dt.tz_convert(TZ)
    df = pd.DataFrame({"day": local.dt.date, "hour": local.dt.hour, "pvpc": prices["pvpc"].values})
    m = df.pivot_table(index="day", columns="hour", values="pvpc", aggfunc="mean")
    m = m.reindex(columns=HOURS)
    m.index = pd.to_datetime(m.index)
    full_index = pd.date_range(m.index.min(), m.index.max(), freq="D")
    m = m.reindex(full_index)
    # Fill isolated gaps (e.g. the missing DST hour) along the hour axis only.
    n_missing = m.isna().sum(axis=1)
    small_gap = (n_missing > 0) & (n_missing <= 2)
    m.loc[small_gap] = m.loc[small_gap].interpolate(axis=1, limit_direction="both")
    m.columns = HOURS
    m.index.name = "day"
    return m


def complete_days(matrix: pd.DataFrame) -> pd.DatetimeIndex:
    return matrix.index[matrix.notna().all(axis=1)]


def _offday(days: pd.DatetimeIndex) -> np.ndarray:
    return np.array([(d.dayofweek >= 5) or is_holiday(d.date()) for d in days], dtype=int)


def build_features(matrix: pd.DataFrame, k: int) -> pd.DataFrame:
    """Long table with one row per (day, hour) and the target ``y``; ``y`` may be NaN for future days."""
    if k < 1 or k > 7:
        raise ValueError("k must be between 1 and 7 days")
    lag = matrix.shift(k)          # day D-k, the most recent day known at issue time
    week = matrix.shift(7)         # same weekday one week before
    lag_stats = pd.DataFrame(
        {
            "lag_mean": lag.mean(axis=1),
            "lag_min": lag.min(axis=1),
            "lag_max": lag.max(axis=1),
            "lag_std": lag.std(axis=1),
            "week_mean": week.mean(axis=1),
            # Mean of the 7 days ending at D-k: captures the slow-moving level (gas, hydro...).
            "trend_mean": matrix.mean(axis=1).rolling(7, min_periods=4).mean().shift(k),
        },
        index=matrix.index,
    )
    days = matrix.index
    offday = pd.Series(_offday(days), index=days)

    rows = []
    for h in HOURS:
        part = pd.DataFrame(index=days)
        part["hour"] = h
        part["y"] = matrix[h]
        part["lag_h"] = lag[h]
        part["lag_h_prev"] = lag[(h - 1) % 24]
        part["lag_h_next"] = lag[(h + 1) % 24]
        part["week_h"] = week[h]
        rows.append(part)
    long = pd.concat(rows)
    long = long.join(lag_stats)
    long["dow"] = long.index.dayofweek
    long["month"] = long.index.month
    doy = long.index.dayofyear
    long["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    long["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    long["is_offday"] = offday.reindex(long.index).values
    long["lag_offday"] = offday.shift(k).reindex(long.index).values
    long.index.name = "day"
    long = long.reset_index().sort_values(["day", "hour"]).reset_index(drop=True)
    return long


def extend_to(matrix: pd.DataFrame, last_day: pd.Timestamp) -> pd.DataFrame:
    """Append empty rows up to ``last_day`` so that future days get feature rows."""
    if matrix.index.max() >= last_day:
        return matrix
    idx = pd.date_range(matrix.index.min(), last_day, freq="D")
    return matrix.reindex(idx)
