from __future__ import annotations

import os
import sys
import pandas as pd
import yaml

sys.path.append(os.path.abspath("."))

from src.common.logging import setup_logger
from src.backtest.simple_backtest import BacktestParams, run_backtest, max_drawdown, annualized_sharpe


def main() -> None:
    logger = setup_logger("run_backtest_ema_ratio")

    with open("config/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    symbol = cfg["symbol"]
    interval = cfg["timebars"]["interval"]
    processed_dir = cfg["paths"]["processed_dir"]

    in_path = f"{processed_dir}/{symbol}_{interval}_ema_ratio_positions.csv"
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Missing positions dataset: {in_path}. Run scripts/compute_positions_ema_ratio.py first.")

    df = pd.read_csv(in_path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").set_index("open_time")

    df = df.rename(columns={"position_ema_ratio": "position"})

    bt_params = BacktestParams(cost_per_turnover=0.0002, execution_lag=1)
    bt = run_backtest(df, bt_params)

    bars_per_year = 365 * 24 * 4
    sharpe_g = annualized_sharpe(bt["strat_ret_gross"], bars_per_year)
    sharpe_n = annualized_sharpe(bt["strat_ret_net"], bars_per_year)
    mdd_g = max_drawdown(bt["equity_gross"])
    mdd_n = max_drawdown(bt["equity_net"])

    logger.info(f"Sharpe gross: {sharpe_g:.3f} | Sharpe net: {sharpe_n:.3f}")
    logger.info(f"Max DD gross: {mdd_g:.3%} | Max DD net: {mdd_n:.3%}")
    logger.info(f"Total turnover: {bt['turnover'].sum():.2f}")
    logger.info(f"Pct time in market: {(bt['pos_exec'].abs() > 0).mean():.3f}")

    out_path = f"{processed_dir}/{symbol}_{interval}_backtest_ema_ratio.csv"
    bt.to_csv(out_path)
    logger.info(f"Saved backtest output: {out_path}")
    logger.info(f"Tail equity:\n{bt[['equity_gross','equity_net']].tail(3)}")


if __name__ == "__main__":
    main()
