from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


PositionMode = Literal["long_only", "long_short"]


@dataclass(frozen=True)
class MACDTrendStrategyParams:
    # Use MACD - Signal spread (hist) for decisions
    entry_threshold: float = 0.0   # enter when hist crosses +thr or -thr
    exit_threshold: float = 0.0    # exit when hist crosses back inside (-thr_exit, +thr_exit)
    mode: PositionMode = "long_short"
    confirm_bars: int = 1 
    cooldown_bars: int = 0


def _confirm_condition(x: np.ndarray, t: int, k: int, predicate) -> bool:
    """
    Require predicate(x[i]) holds for i=t-k+1..t (k bars), with bounds checking.
    """
    start = max(0, t - k + 1)
    window = x[start:t + 1]
    if len(window) < k:
        return False
    return bool(np.all([predicate(v) for v in window]))


def generate_positions_from_macd(
    macd_df: pd.DataFrame,
    params: MACDTrendStrategyParams,
) -> pd.Series:
    """
    Inputs
    ------
    macd_df: DataFrame with columns ["hist_norm", "close", "ema_slow"] indexed by time
    params: strategy params

    Output
    ------
    position: pd.Series in {-1, 0, +1}
    """
    required = {"close", "ema_slow", "hist_norm"}
    missing = required.difference(set(macd_df.columns))
    if missing:
        raise ValueError(f"macd_df missing columns: {missing}")

    close = macd_df["close"].astype(float).to_numpy()
    ema_slow = macd_df["ema_slow"].astype(float).to_numpy()
    hist = macd_df["hist_norm"].astype(float).to_numpy()
    n = len(hist)
    pos = np.zeros(n, dtype=int)

    entry_thr = float(params.entry_threshold)
    exit_thr = float(params.exit_threshold)
    if exit_thr > entry_thr:
        raise ValueError("exit_threshold should be <= entry_threshold (hysteresis band)")

    confirm = int(params.confirm_bars)
    if confirm <= 0:
        raise ValueError("confirm_bars must be >= 1")

    cooldown = int(params.cooldown_bars)
    if cooldown < 0:
        raise ValueError("cooldown_bars must be >= 0")

    current = 0
    cooldown_left = 0

    for t in range(n):
        h = hist[t]
        if not np.isfinite(h):
            pos[t] = current
            continue
        
        # Regime filter: gate longs/shorts based on slow EMA state
        regime_long_ok = (
            np.isfinite(close[t]) and np.isfinite(ema_slow[t]) and (close[t] >= ema_slow[t])
        )
        regime_short_ok = (
            np.isfinite(close[t]) and np.isfinite(ema_slow[t]) and (close[t] <= ema_slow[t])
        )

        if cooldown_left > 0:
            cooldown_left -= 1
            pos[t] = current
            continue
        
        # If regime flips against current position, exit to flat
        if current == 1 and not regime_long_ok:
            current = 0
            cooldown_left = cooldown
            pos[t] = current
            continue
        
        if current == -1 and not regime_short_ok:
            current = 0
            cooldown_left = cooldown
            pos[t] = current
            continue
        
        # --- Exit rules (apply first) ---
        if current == 1:
            # exit long when hist <= +exit_thr
            if _confirm_condition(hist, t, confirm, lambda v: np.isfinite(v) and v <= exit_thr):
                current = 0
                cooldown_left = cooldown
        
        elif current == -1:
            # exit short when hist >= -exit_thr
            if _confirm_condition(hist, t, confirm, lambda v: np.isfinite(v) and v >= -exit_thr):
                current = 0
                cooldown_left = cooldown

        # --- Entry / Flip rules ---
        if current == 0:
        # Enter long only if regime_long_ok
            if regime_long_ok and _confirm_condition(hist, t, confirm, lambda v: np.isfinite(v) and v >= entry_thr):
                current = 1
                cooldown_left = cooldown
            # Enter short only if regime_short_ok (and long_short mode)
            
            elif params.mode == "long_short":
                if regime_short_ok and _confirm_condition(hist, t, confirm, lambda v: np.isfinite(v) and v <= -entry_thr):
                    current = -1
                    cooldown_left = cooldown
        
        # elif current == 1 and params.mode == "long_short":
        #     # flip long->short when strong negative signal
        #     if _confirm_condition(hist, t, confirm, lambda v: np.isfinite(v) and v <= -entry_thr):
        #         current = -1
        #         cooldown_left = cooldown

        # elif current == -1:
        #     # flip short->long when strong positive signal
        #     if _confirm_condition(hist, t, confirm, lambda v: np.isfinite(v) and v >= entry_thr):
        #         current = 1
        #         cooldown_left = cooldown

        pos[t] = current

    return pd.Series(pos, index=macd_df.index, name="position")
