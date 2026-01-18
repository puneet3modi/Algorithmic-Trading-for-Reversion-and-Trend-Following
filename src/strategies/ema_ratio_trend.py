from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EMARatioTrendParams:
    entry_threshold: float = 0.0010   # 10 bps
    exit_threshold: float = 0.0004    # 4 bps
    confirm_bars: int = 2
    cooldown_bars: int = 1
    mode: str = "long_short"  # "long_only" or "long_short"


def _confirm(series: np.ndarray, t: int, k: int, predicate) -> bool:
    if k <= 1:
        return predicate(series[t])
    start = max(0, t - k + 1)
    for i in range(start, t + 1):
        if not predicate(series[i]):
            return False
    return True


def generate_positions_from_ema_ratio(df: pd.DataFrame, params: EMARatioTrendParams) -> pd.Series:
    required = {"ema_ratio"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"df missing columns: {missing}")

    x = df["ema_ratio"].astype(float).to_numpy()
    n = len(x)

    entry = float(params.entry_threshold)
    exit_ = float(params.exit_threshold)
    if exit_ >= entry:
        raise ValueError("exit_threshold must be < entry_threshold")

    confirm = int(params.confirm_bars)
    cooldown = int(params.cooldown_bars)

    current = 0
    cooldown_left = 0
    pos = np.zeros(n, dtype=int)

    for t in range(n):
        if cooldown_left > 0:
            cooldown_left -= 1
            pos[t] = current
            continue

        v = x[t]
        if not np.isfinite(v):
            pos[t] = current
            continue

        # Exit logic: go flat inside deadband
        if current == 1 and v <= exit_:
            current = 0
            cooldown_left = cooldown
        elif current == -1 and v >= -exit_:
            current = 0
            cooldown_left = cooldown

        # Entry logic
        if current == 0:
            if _confirm(x, t, confirm, lambda z: np.isfinite(z) and z >= entry):
                current = 1
                cooldown_left = cooldown
            elif params.mode == "long_short":
                if _confirm(x, t, confirm, lambda z: np.isfinite(z) and z <= -entry):
                    current = -1
                    cooldown_left = cooldown

        pos[t] = current

    return pd.Series(pos, index=df.index, name="position_ema_ratio")
