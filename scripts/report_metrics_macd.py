from __future__ import annotations

import os
import sys

import pandas as pd
import yaml

sys.path.append(os.path.abspath("."))

from src.common.logging import setup_logger
from src.backtest.simple_backtest import max_drawdown, annualized_sharpe


def main() -> None:
    logger = setup_logger("report_metrics_macd")

    with open("config/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    symbol = cfg["symbol"]
    interval = cfg["timebars"]["interval"]
    processed_dir = cfg["paths"]["processed_dir"]

    path = f"{processed_dir}/{symbol}_{interval}_backtest_macd.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing backtest output: {path}")

    bt = pd.read_csv(path)
    bt["open_time"] = pd.to_datetime(bt["open_time"], utc=True)
    bt = bt.sort_values("open_time").set_index("open_time")

    bars_per_year = 365 * 24 * 4

    metrics = {
        "period_start": str(bt.index.min()),
        "period_end": str(bt.index.max()),
        "final_equity_gross": float(bt["equity_gross"].iloc[-1]),
        "final_equity_net": float(bt["equity_net"].iloc[-1]),
        "sharpe_gross": annualized_sharpe(bt["strat_ret_gross"], bars_per_year),
        "sharpe_net": annualized_sharpe(bt["strat_ret_net"], bars_per_year),
        "max_dd_gross": max_drawdown(bt["equity_gross"]),
        "max_dd_net": max_drawdown(bt["equity_net"]),
        "total_turnover": float(bt["turnover"].sum()),
        "avg_turnover_per_bar": float(bt["turnover"].mean()),
        "pct_time_in_market": float((bt["pos_exec"].abs() > 0).mean()),
    }

    out = pd.DataFrame([metrics])
    out_path = f"{processed_dir}/{symbol}_{interval}_metrics_macd.csv"
    out.to_csv(out_path, index=False)

    logger.info(f"Saved metrics: {out_path}")
    logger.info(f"\n{out.to_string(index=False)}")


if __name__ == "__main__":
    main()
    