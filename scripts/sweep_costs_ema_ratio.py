from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import yaml

sys.path.append(os.path.abspath("."))

from src.common.logging import setup_logger
from src.backtest.simple_backtest import BacktestParams, run_backtest, max_drawdown, annualized_sharpe


def main() -> None:
    logger = setup_logger("sweep_costs_ema_ratio")

    with open("config/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    symbol = cfg["symbol"]
    interval = cfg["timebars"]["interval"]
    processed_dir = cfg["paths"]["processed_dir"]

    in_path = f"{processed_dir}/{symbol}_{interval}_ema_ratio_positions.csv"
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Missing positions dataset: {in_path}")

    df = pd.read_csv(in_path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").set_index("open_time")
    df = df.rename(columns={"position_ema_ratio": "position"})

    bars_per_year = 365 * 24 * 4

    grid_bps = [0, 0.5, 1, 2, 3, 5, 7.5, 10]
    rows = []

    for bps in grid_bps:
        cost = bps / 10000.0
        bt_params = BacktestParams(cost_per_turnover=cost, execution_lag=1)
        bt = run_backtest(df, bt_params)

        rows.append(
            {
                "cost_bps_per_turnover": bps,
                "sharpe_gross": annualized_sharpe(bt["strat_ret_gross"], bars_per_year),
                "sharpe_net": annualized_sharpe(bt["strat_ret_net"], bars_per_year),
                "mdd_gross": max_drawdown(bt["equity_gross"]),
                "mdd_net": max_drawdown(bt["equity_net"]),
                "final_equity_gross": float(bt["equity_gross"].iloc[-1]),
                "final_equity_net": float(bt["equity_net"].iloc[-1]),
                "total_turnover": float(bt["turnover"].sum()),
            }
        )

    res = pd.DataFrame(rows)
    out_path = f"{processed_dir}/{symbol}_{interval}_cost_sweep_ema_ratio.csv"
    res.to_csv(out_path, index=False)

    logger.info(f"Saved cost sweep: {out_path}")
    logger.info(f"\n{res.to_string(index=False)}")


if __name__ == "__main__":
    main()
