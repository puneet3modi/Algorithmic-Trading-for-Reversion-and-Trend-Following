from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QAConfig:
    expected_interval_minutes: int = 15
    max_abs_log_return: float = 0.35
    outlier_rolling_window: int = 96
    outlier_sigma: float = 10.0


def _expected_time_index(df: pd.DataFrame, freq_minutes: int) -> pd.DatetimeIndex:
    start = df["open_time"].min()
    end = df["open_time"].max()
    return pd.date_range(start=start, end=end, freq=f"{freq_minutes}min", tz="UTC")


def run_qa(df: pd.DataFrame, cfg: QAConfig) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = df.copy()

    # If open_time is a string (CSV load), convert to datetime UTC
    if not np.issubdtype(df["open_time"].dtype, np.datetime64):
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")

    # Sort and drop duplicates
    df = df.sort_values("open_time").reset_index(drop=True)
    duplicates = int(df["open_time"].duplicated().sum())
    df = df.drop_duplicates(subset=["open_time"]).reset_index(drop=True)

    monotonic = bool(df["open_time"].is_monotonic_increasing)

    # Structural validity checks
    neg_or_zero_prices = int(((df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0) | (df["close"] <= 0)).sum())
    neg_volume = int((df["volume"] < 0).sum())

    # Missing bars against expected grid
    expected_idx = _expected_time_index(df, cfg.expected_interval_minutes)
    actual_idx = pd.DatetimeIndex(df["open_time"])
    missing = expected_idx.difference(actual_idx)
    missing_count = int(len(missing))
    missing_pct = float(missing_count / max(len(expected_idx), 1))

    # Outliers: compute log returns
    df["log_close"] = np.log(df["close"].astype(float))
    df["logret"] = df["log_close"].diff()

    df["flag_abs_logret"] = df["logret"].abs() > cfg.max_abs_log_return

    rolling_std = df["logret"].rolling(
        cfg.outlier_rolling_window,
        min_periods=max(5, cfg.outlier_rolling_window // 5),
    ).std()

    df["flag_sigma_outlier"] = df["logret"].abs() > (cfg.outlier_sigma * rolling_std)

    outliers_abs = int(df["flag_abs_logret"].sum(skipna=True))
    outliers_sigma = int(df["flag_sigma_outlier"].sum(skipna=True))

    summary: Dict[str, object] = {
        "rows": int(len(df)),
        "start_open_time_utc": str(df["open_time"].min()),
        "end_open_time_utc": str(df["open_time"].max()),
        "duplicates_removed": duplicates,
        "monotonic_increasing": monotonic,
        "neg_or_zero_prices": neg_or_zero_prices,
        "neg_volume": neg_volume,
        "missing_bars_count": missing_count,
        "missing_bars_pct": missing_pct,
        "outliers_abslogret_count": outliers_abs,
        "outliers_sigma_count": outliers_sigma,
    }

    return df, summary


def summary_to_df(summary: Dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame([summary])
