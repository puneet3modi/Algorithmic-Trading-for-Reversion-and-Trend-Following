from __future__ import annotations

import os
import sys

import pandas as pd
import yaml

sys.path.append(os.path.abspath("."))

from src.common.logging import setup_logger
from src.strategies.macd_trend import MACDTrendStrategyParams, generate_positions_from_macd


def main() -> None:
    logger = setup_logger("compute_positions_macd")

    with open("config/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    symbol = cfg["symbol"]
    interval = cfg["timebars"]["interval"]
    processed_dir = cfg["paths"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    in_path = f"{processed_dir}/{symbol}_{interval}_with_macd.csv"
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Indicators dataset not found: {in_path}. Run scripts/compute_indicators.py first.")

    df = pd.read_csv(in_path)
    logger.info(f"Norm columns present: {[c for c in df.columns if 'norm' in c]}")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").set_index("open_time")

    macd_df = df[["close", "ema_slow", "hist_norm"]]

    params = MACDTrendStrategyParams(
        entry_threshold=0.0007,
        exit_threshold=0.0003,
        mode="long_short",
        confirm_bars=3,
        cooldown_bars=2,
    )

    pos = generate_positions_from_macd(macd_df, params)
    df["position"] = pos

    out_path = f"{processed_dir}/{symbol}_{interval}_with_macd_positions.csv"
    df.to_csv(out_path)
    logger.info(f"Saved positions dataset: {out_path}")

    # Quick summary
    counts = df["position"].value_counts(dropna=False).sort_index()
    logger.info(f"Position counts:\n{counts}")
    logger.info(f"Tail:\n{df[['close','macd','signal','hist','macd_norm','signal_norm','hist_norm','position']].tail(5)}")


if __name__ == "__main__":
    main()
