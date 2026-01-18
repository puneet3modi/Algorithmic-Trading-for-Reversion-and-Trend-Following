from __future__ import annotations

import os
import sys

import pandas as pd
import yaml

sys.path.append(os.path.abspath("."))

from src.common.logging import setup_logger
from src.data.fetch_klines import make_output_paths
from src.data.quality import QAConfig, run_qa, summary_to_df


def main() -> None:
    logger = setup_logger("run_qa")

    with open("config/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    symbol = cfg["symbol"]
    interval = cfg["timebars"]["interval"]
    raw_dir = cfg["paths"]["raw_dir"]
    processed_dir = cfg["paths"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    csv_path = make_output_paths(raw_dir, symbol, interval)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw data not found: {csv_path}. Run scripts/fetch_data.py first.")

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded raw: {len(df)} rows")

    qa_cfg = QAConfig(
        expected_interval_minutes=int(cfg["qa"]["expected_interval_minutes"]),
        max_abs_log_return=float(cfg["qa"]["max_abs_log_return"]),
        outlier_rolling_window=int(cfg["qa"]["outlier_rolling_window"]),
        outlier_sigma=float(cfg["qa"]["outlier_sigma"]),
    )

    df2, summary = run_qa(df, qa_cfg)

    # Save processed dataset (still raw-ish, but QA flags included)
    processed_path = f"{processed_dir}/{symbol}_{interval}_klines_processed.csv"
    df2.to_csv(processed_path, index=False)

    # Save QA summary
    summary_df = summary_to_df(summary)
    qa_summary_path = f"{processed_dir}/qa_summary_{symbol}_{interval}.csv"
    summary_df.to_csv(qa_summary_path, index=False)

    logger.info(f"Saved processed parquet: {processed_path}")
    logger.info(f"Saved QA summary: {qa_summary_path}")
    logger.info(f"QA summary: {summary}")


if __name__ == "__main__":
    main()
