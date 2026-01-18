from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ShockReversionParams:
    k_entry: float = 2.0
    k_exit: float = 0.5
    trend_gate: float = 0.0010
    max_hold_bars: int = 16
    cooldown_bars: int = 1
    mode: str = "long_short"


def generate_positions_shock_reversion(df: pd.DataFrame, params: ShockReversionParams) -> pd.Series:
    required = {"shock", "ema_ratio"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"df missing columns: {missing}")

    shock = df["shock"].astype(float).to_numpy()
    trend = df["ema_ratio"].astype(float).to_numpy()
    n = len(df)

    current = 0
    cooldown_left = 0
    hold = 0
    pos = np.zeros(n, dtype=int)

    for t in range(n):
        if cooldown_left > 0:
            cooldown_left -= 1
            pos[t] = current
            continue

        s = shock[t]
        tr = trend[t]
        if not (np.isfinite(s) and np.isfinite(tr)):
            pos[t] = current
            continue

        gate_ok = abs(tr) <= params.trend_gate

        # manage holding time
        if current != 0:
            hold += 1

        # exits
        if current == 1:
            if s >= -params.k_exit or hold >= params.max_hold_bars:
                current = 0
                hold = 0
                cooldown_left = params.cooldown_bars

        elif current == -1:
            if s <= params.k_exit or hold >= params.max_hold_bars:
                current = 0
                hold = 0
                cooldown_left = params.cooldown_bars

        # entries
        if current == 0 and gate_ok:
            if s <= -params.k_entry:
                current = 1
                hold = 0
                cooldown_left = params.cooldown_bars
            elif params.mode == "long_short" and s >= params.k_entry:
                current = -1
                hold = 0
                cooldown_left = params.cooldown_bars

        pos[t] = current

    return pd.Series(pos, index=df.index, name="position_reversion")
