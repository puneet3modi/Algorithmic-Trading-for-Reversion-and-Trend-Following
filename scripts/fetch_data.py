from __future__ import annotations

import os
import sys
from venv import logger

import yaml

# Allow "src" imports when running as a script
sys.path.append(os.path.abspath("."))

from src.common.logging import setup_logger
from src.data.binance_public import BinancePublicClient
from src.data.fetch_klines import (
    KlinesFetchSpec,
    fetch_klines_to_df,
    make_output_paths,
    save_df,
)


def main() -> None:
    logger = setup_logger("fetch_data")

    with open("config/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    symbol = cfg["symbol"]
    interval = cfg["timebars"]["interval"]
    start_utc = cfg["timebars"]["start_utc"]
    end_utc = cfg["timebars"]["end_utc"]
    limit = int(cfg["timebars"].get("limit_per_request", 1000))

    raw_dir = cfg["paths"]["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)

    spec = KlinesFetchSpec(
        symbol=symbol,
        interval=interval,
        start_utc=start_utc,
        end_utc=end_utc,
        limit_per_request=limit,
    )

    client = BinancePublicClient()

    logger.info(f"Fetching klines: {symbol} {interval} from {start_utc} to {end_utc}")
    df = fetch_klines_to_df(client, spec)
    logger.info(
        f"Fetched rows: {len(df)} | "
        f"time range: {df['open_time'].min()} -> {df['open_time'].max()}"
    )

    csv_path = make_output_paths(raw_dir, symbol, interval)
    save_df(df, csv_path=csv_path)
    logger.info(f"Saved raw CSV: {csv_path}")


if __name__ == "__main__":
    main()
