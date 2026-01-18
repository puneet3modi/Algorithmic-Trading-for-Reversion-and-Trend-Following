from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from src.indicators.ema import ema, EMAParams


@dataclass(frozen=True)
class EMARatioParams:
    fast: int = 20
    slow: int = 100
    init: str = "sma"
    min_periods: int | None = None


def ema_ratio(close: pd.Series, params: EMARatioParams) -> pd.DataFrame:
    if params.fast <= 0 or params.slow <= 0:
        raise ValueError("EMA periods must be positive")
    if params.fast >= params.slow:
        raise ValueError("Requires fast < slow")

    min_p = params.min_periods if params.min_periods is not None else params.slow

    ema_fast = ema(close, EMAParams(span=params.fast, init=params.init, min_periods=min_p)).rename("ema_fast_2")
    ema_slow = ema(close, EMAParams(span=params.slow, init=params.init, min_periods=min_p)).rename("ema_slow_2")

    ratio = (ema_fast / ema_slow - 1.0).rename("ema_ratio")

    return pd.concat([ema_fast, ema_slow, ratio], axis=1)
