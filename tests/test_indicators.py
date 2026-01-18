import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath("."))

from src.indicators.ema import EMAParams, ema
from src.indicators.macd import MACDParams, macd


def test_ema_constant_series():
    idx = pd.date_range("2024-01-01", periods=200, freq="15min", tz="UTC")
    s = pd.Series(100.0, index=idx)
    e = ema(s, EMAParams(span=20, init="price"))
    tail = e.dropna().tail(50)
    assert np.allclose(tail.values, 100.0, atol=1e-10)


def test_macd_constant_series_near_zero():
    idx = pd.date_range("2024-01-01", periods=500, freq="15min", tz="UTC")
    s = pd.Series(250.0, index=idx)
    m = macd(s, MACDParams(fast=12, slow=26, signal=9, init="price"))
    tail = m.dropna().tail(50)
    assert np.allclose(tail["macd"].values, 0.0, atol=1e-10)
    assert np.allclose(tail["hist"].values, 0.0, atol=1e-10)


def test_macd_fast_less_than_slow():
    idx = pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC")
    s = pd.Series(np.linspace(100, 200, 100), index=idx)
    try:
        _ = macd(s, MACDParams(fast=26, slow=12, signal=9))
        assert False, "Expected ValueError"
    except ValueError:
        assert True
