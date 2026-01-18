from __future__ import annotations

import os
import sys

import pandas as pd
import yaml

sys.path.append(os.path.abspath("."))

from src.common.logging import setup_logger
from src.data.fetch_klines import make_output_paths  # your CSV path helper
from src.indicators.macd import MACDParams, macd


def main() -> None:
    logger = setup_logger("compute_indicators")

    with open("config/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    symbol = cfg["symbol"]
    interval = cfg["timebars"]["interval"]
    raw_dir = cfg["paths"]["raw_dir"]
    processed_dir = cfg["paths"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    csv_path = make_output_paths(raw_dir, symbol, interval)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw CSV not found: {csv_path}. Run scripts/fetch_data.py first.")

    df = pd.read_csv(csv_path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").set_index("open_time")

    close = df["close"].astype(float)

    m = macd(close, MACDParams(fast=12, slow=26, signal=9, init="price"))
    out = pd.concat([df[["open", "high", "low", "close", "volume"]], m], axis=1)

    out_path = f"{processed_dir}/{symbol}_{interval}_with_macd.csv"
    out.to_csv(out_path)
    logger.info(f"Saved indicators dataset: {out_path}")
    logger.info(f"Tail:\n{out.tail(3)}")


if __name__ == "__main__":
    main()
