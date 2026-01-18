from __future__ import annotations

import numpy as np
import pandas as pd


def add_basic_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assumes df indexed by open_time and has at least: close, volume.
    Uses existing columns if present: hist_norm, macd_norm, signal_norm, ema_ratio, dist, ewma_vol.
    """
    out = df.copy()

    close = out["close"].astype(float)
    volu = out["volume"].astype(float)

    out["logret"] = np.log(close).diff()

    # Rolling return stats (stationary-ish)
    out["ret_mean_32"] = out["logret"].rolling(32).mean()
    out["ret_std_32"] = out["logret"].rolling(32).std()

    # Volume normalization
    out["vol_z_96"] = (volu - volu.rolling(96).mean()) / (volu.rolling(96).std() + 1e-12)

    # Momentum-ish
    out["mom_32"] = close.pct_change(32)
    out["mom_96"] = close.pct_change(96)

    return out
