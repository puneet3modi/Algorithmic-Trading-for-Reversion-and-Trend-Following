from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_vwap(close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """
    Rolling VWAP over bars using close as price proxy:
    VWAP_t = sum(P_i V_i)/sum(V_i) over the last `window` bars.
    """
    p = close.astype(float)
    v = volume.astype(float)

    pv = (p * v).rolling(window=window, min_periods=window).sum()
    vv = v.rolling(window=window, min_periods=window).sum()

    vwap = pv / vv.replace(0.0, np.nan)
    return vwap.rename(f"vwap_{window}")
