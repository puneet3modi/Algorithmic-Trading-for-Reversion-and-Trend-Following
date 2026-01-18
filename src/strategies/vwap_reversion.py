from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VWAPReversionParams:
    vwap_window: int = 96
    k_entry: float = 2.0
    k_exit: float = 0.5
    max_hold_bars: int = 16
    cooldown_bars: int = 1
    trend_gate: float = 0.0020     # NEW: only trade if abs(ema_ratio) <= trend_gate
    stop_k: float = 4.0            # NEW: stop out if |dist| > stop_k * vol
    mode: str = "long_short"


def generate_positions_vwap_reversion(df: pd.DataFrame, params: VWAPReversionParams) -> pd.Series:
    required = {"dist", "ewma_vol", "ema_ratio"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"df missing columns: {missing}")

    dist = df["dist"].astype(float).to_numpy()
    vol = df["ewma_vol"].astype(float).to_numpy()
    tr = df["ema_ratio"].astype(float).to_numpy()

    n = len(df)
    current = 0
    hold = 0
    cooldown = 0
    pos = np.zeros(n, dtype=int)

    for t in range(n):
        if cooldown > 0:
            cooldown -= 1
            pos[t] = current
            continue

        d = dist[t]
        s = vol[t]
        trend = tr[t]

        if not (np.isfinite(d) and np.isfinite(s) and s > 0 and np.isfinite(trend)):
            pos[t] = current
            continue

        entry = params.k_entry * s
        exitb = params.k_exit * s
        stopb = params.stop_k * s

        if current != 0:
            hold += 1

        # stop-outs (risk control)
        if current == 1 and d < -stopb:
            current = 0
            hold = 0
            cooldown = params.cooldown_bars

        elif current == -1 and d > stopb:
            current = 0
            hold = 0
            cooldown = params.cooldown_bars

        # normal exits
        if current == 1:
            if d >= -exitb or hold >= params.max_hold_bars:
                current = 0
                hold = 0
                cooldown = params.cooldown_bars

        elif current == -1:
            if d <= exitb or hold >= params.max_hold_bars:
                current = 0
                hold = 0
                cooldown = params.cooldown_bars

        # entries ONLY in low-trend regime
        if current == 0 and np.abs(trend) <= params.trend_gate:
            if d <= -entry:
                current = 1
                hold = 0
                cooldown = params.cooldown_bars
            elif params.mode == "long_short" and d >= entry:
                current = -1
                hold = 0
                cooldown = params.cooldown_bars

        pos[t] = current

    return pd.Series(pos, index=df.index, name="position_reversion_vwap")
