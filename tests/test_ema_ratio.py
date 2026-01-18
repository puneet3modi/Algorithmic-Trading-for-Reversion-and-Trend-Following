import numpy as np
import pandas as pd
import pytest

from src.indicators.ema_ratio import EMARatioParams, ema_ratio
from src.strategies.ema_ratio_trend import EMARatioTrendParams, generate_positions_from_ema_ratio


def test_ema_ratio_columns():
    s = pd.Series(np.linspace(100, 200, 300))
    out = ema_ratio(s, EMARatioParams(fast=10, slow=50))
    assert "ema_ratio" in out.columns
    assert len(out) == len(s)


def test_ema_ratio_trend_positions_basic():
    idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    df = pd.DataFrame({"ema_ratio": [0, 0, 0.002, 0.002, 0.002, 0.0001, -0.002, -0.002, -0.0001, 0]}, index=idx)

    params = EMARatioTrendParams(entry_threshold=0.001, exit_threshold=0.0004, confirm_bars=2, cooldown_bars=0)
    pos = generate_positions_from_ema_ratio(df, params)
    
    assert pos.max() == 1
    assert pos.min() == -1
#   Expect entry at index 3 (2 bars above 0.001), exit at index 5 (<=0.0004),