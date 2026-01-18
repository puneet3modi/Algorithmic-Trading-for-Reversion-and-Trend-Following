from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.indicators.ema import EMAParams, ema


@dataclass(frozen=True)
class MACDParams:
    fast: int = 12
    slow: int = 26
    signal: int = 9
    init: str = "price"          # passed through to EMA init
    min_periods: Optional[int] = None  # if None, uses slow


def macd(close: pd.Series, params: MACDParams) -> pd.DataFrame:
    """
    Compute MACD:
    macd = EMA_fast(close) - EMA_slow(close)
    signal = EMA_signal(macd)
    hist = macd - signal

    Also returns normalized variants:
    macd_norm = macd / ema_slow
    signal_norm = signal / ema_slow
    hist_norm = hist / ema_slow
    """
    if params.fast <= 0 or params.slow <= 0 or params.signal <= 0:
        raise ValueError("MACD periods must be positive")
    if params.fast >= params.slow:
        raise ValueError("MACD requires fast < slow")

    min_p = params.min_periods if params.min_periods is not None else params.slow

    ema_fast = ema(close, EMAParams(span=params.fast, init=params.init, min_periods=min_p)).rename("ema_fast")
    ema_slow = ema(close, EMAParams(span=params.slow, init=params.init, min_periods=min_p)).rename("ema_slow")

    macd_line = (ema_fast - ema_slow).rename("macd")
    signal_line = ema(macd_line, EMAParams(span=params.signal, init=params.init, min_periods=min_p)).rename("signal")
    hist = (macd_line - signal_line).rename("hist")

    # Normalization (avoid divide by zero)
    denom = ema_slow.replace(0.0, pd.NA).astype(float)

    macd_norm = (macd_line / denom).rename("macd_norm")
    signal_norm = (signal_line / denom).rename("signal_norm")
    hist_norm = (hist / denom).rename("hist_norm")

    return pd.concat(
        [ema_fast, ema_slow, macd_line, signal_line, hist, macd_norm, signal_norm, hist_norm],
        axis=1,
    )
