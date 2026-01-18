import numpy as np
import pandas as pd

from src.strategies.macd_trend import MACDTrendStrategyParams, generate_positions_from_macd


def test_strategy_hysteresis_basic():
    idx = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")

    # Create a toy histogram series and treat it as "hist_norm" for unit testing.
    hist_norm = np.array([0, 0.2, 0.9, 1.1, 1.2, 0.8, 0.4, 0.1, 0.0, -0.2], dtype=float)

    # Dummy close/ema_slow to satisfy the regime-gate inputs (keep close > ema_slow to allow trading)
    close = np.full(len(idx), 100.0)
    ema_slow = np.full(len(idx), 99.0)

    df = pd.DataFrame(
        {"hist_norm": hist_norm, "close": close, "ema_slow": ema_slow},
        index=idx,
    )

    params = MACDTrendStrategyParams(entry_threshold=1.0, exit_threshold=0.3, mode="long_only", confirm_bars=1)
    pos = generate_positions_from_macd(df, params)

    # Expect: enters long when hist_norm > entry_threshold and exits when <= exit_threshold
    assert pos.iloc[0] == 0
    assert pos.iloc[3] == 1  # crosses above 1.0
    assert pos.iloc[6] == 1  # still above exit threshold
    assert pos.iloc[7] == 0  # drops to 0.1 <= 0.3 -> exit


def test_strategy_confirm_bars():
    idx = pd.date_range("2024-01-01", periods=7, freq="15min", tz="UTC")
    hist_norm = np.array([0.0, 1.1, 0.9, 1.2, 1.3, 0.2, 0.2])

    close = np.full(len(idx), 100.0)
    ema_slow = np.full(len(idx), 99.0)

    df = pd.DataFrame(
        {"hist_norm": hist_norm, "close": close, "ema_slow": ema_slow},
        index=idx,
    )

    params = MACDTrendStrategyParams(
        entry_threshold=1.0,
        exit_threshold=0.3,
        mode="long_only",
        confirm_bars=2,
    )
    pos = generate_positions_from_macd(df, params)

    # confirm_bars=2 means we need 2 consecutive bars above entry_threshold to enter
    # sequence above 1.0 occurs at indices 3 and 4 -> enter at 4
    assert pos.iloc[0] == 0
    assert pos.iloc[3] == 0
    assert pos.iloc[4] == 1

    # Exit may apply on the *next* bar depending on strategy update convention
    assert pos.iloc[5] == 1
    assert pos.iloc[6] == 0
    