from __future__ import annotations

import os
import sys
import pandas as pd
import yaml

sys.path.append(os.path.abspath("."))

from src.common.logging import setup_logger
from src.indicators.ema_ratio import EMARatioParams, ema_ratio
from src.strategies.ema_ratio_trend import EMARatioTrendParams, generate_positions_from_ema_ratio


def main() -> None:
    logger = setup_logger("compute_positions_ema_ratio")

    with open("config/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    symbol = cfg["symbol"]
    interval = cfg["timebars"]["interval"]
    processed_dir = cfg["paths"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    in_path = f"{processed_dir}/{symbol}_{interval}_with_macd.csv"
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Missing indicators dataset: {in_path}. Run scripts/compute_indicators.py first.")

    df = pd.read_csv(in_path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").set_index("open_time")

    # Build EMA ratio features (separate from MACD EMA columns)
    ratio_df = ema_ratio(df["close"], EMARatioParams(fast=20, slow=100))
    df = pd.concat([df, ratio_df], axis=1)

    strat_params = EMARatioTrendParams(
        entry_threshold=0.0010,
        exit_threshold=0.0004,
        confirm_bars=2,
        cooldown_bars=1,
        mode="long_short",
    )

    df["position_ema_ratio"] = generate_positions_from_ema_ratio(df[["ema_ratio"]], strat_params)

    out_path = f"{processed_dir}/{symbol}_{interval}_ema_ratio_positions.csv"
    df[["close", "ema_ratio", "position_ema_ratio"]].to_csv(out_path)

    logger.info(f"Saved Strategy 2 positions: {out_path}")
    logger.info(f"Position counts:\n{df['position_ema_ratio'].value_counts(dropna=False)}")
    logger.info(f"Tail:\n{df[['close','ema_ratio','position_ema_ratio']].tail(5)}")


if __name__ == "__main__":
    main()
