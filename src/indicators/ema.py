from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


InitMethod = Literal["price", "sma"]


@dataclass(frozen=True)
class EMAParams:
    span: int
    init: InitMethod = "price"  # "price" uses first value, "sma" uses SMA over first span points
    min_periods: Optional[int] = None  # if None, uses span


def ema(series: pd.Series, params: EMAParams) -> pd.Series:
    """
    Compute EMA via the recursive definition (no TA libs).

    Parameters
    ----------
    series : pd.Series
        Input series (e.g., close prices), indexed by time.
    params : EMAParams
        span: EMA span n (alpha = 2/(n+1))
        init: "price" or "sma"
        min_periods: number of observations required to start returning non-NaN

    Returns
    -------
    pd.Series
        EMA values aligned to the input index.
    """
    if params.span <= 0:
        raise ValueError("EMA span must be positive")

    x = series.astype(float).to_numpy()
    n = len(x)
    if n == 0:
        return pd.Series([], index=series.index, dtype=float)

    alpha = 2.0 / (params.span + 1.0)
    min_p = params.min_periods if params.min_periods is not None else params.span

    out = np.full(n, np.nan, dtype=float)

    # Find first finite observation
    finite_idx = np.flatnonzero(np.isfinite(x))
    if len(finite_idx) == 0:
        return pd.Series(out, index=series.index, name=f"ema_{params.span}")

    first = int(finite_idx[0])

    # Initialize EMA
    if params.init == "price":
        ema_prev = float(x[first])
        start = first
    elif params.init == "sma":
        # SMA over first `span` finite points starting at `first`
        # If insufficient points, fallback to first price
        window = x[first:first + params.span]
        window = window[np.isfinite(window)]
        if len(window) < params.span:
            ema_prev = float(x[first])
            start = first
        else:
            ema_prev = float(window.mean())
            start = first + params.span - 1  # first EMA published at end of initial window
    else:
        raise ValueError("init must be 'price' or 'sma'")

    # Recursive update
    for t in range(start, n):
        xt = x[t]
        if not np.isfinite(xt):
            # propagate previous EMA through missing values
            out[t] = ema_prev
            continue
        ema_prev = alpha * xt + (1.0 - alpha) * ema_prev
        out[t] = ema_prev

    # Enforce min_periods: mask early values
    # We count non-NaN inputs from the first finite observation
    count_non_nan = np.cumsum(np.isfinite(x).astype(int))
    out[count_non_nan < min_p] = np.nan

    return pd.Series(out, index=series.index, name=f"ema_{params.span}")
