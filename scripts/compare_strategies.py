from __future__ import annotations

import os
import sys
import pandas as pd
import yaml

sys.path.append(os.path.abspath("."))

from src.common.logging import setup_logger
from src.backtest.simple_backtest import max_drawdown, annualized_sharpe


def _load_bt(path: str) -> pd.DataFrame:
    bt = pd.read_csv(path)
    bt["open_time"] = pd.to_datetime(bt["open_time"], utc=True)
    bt = bt.sort_values("open_time").set_index("open_time")
    return bt


def _metrics(bt: pd.DataFrame, name: str) -> dict:
    bars_per_year = 365 * 24 * 4
    return {
        "strategy": name,
        "final_equity_gross": float(bt["equity_gross"].iloc[-1]),
        "final_equity_net": float(bt["equity_net"].iloc[-1]),
        "sharpe_gross": annualized_sharpe(bt["strat_ret_gross"], bars_per_year),
        "sharpe_net": annualized_sharpe(bt["strat_ret_net"], bars_per_year),
        "max_dd_gross": max_drawdown(bt["equity_gross"]),
        "max_dd_net": max_drawdown(bt["equity_net"]),
        "total_turnover": float(bt["turnover"].sum()),
        "pct_time_in_market": float((bt["pos_exec"].abs() > 0).mean()),
    }


def main() -> None:
    logger = setup_logger("compare_strategies")

    with open("config/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    symbol = cfg["symbol"]
    interval = cfg["timebars"]["interval"]
    processed_dir = cfg["paths"]["processed_dir"]

    p1 = f"{processed_dir}/{symbol}_{interval}_backtest_macd.csv"
    p2 = f"{processed_dir}/{symbol}_{interval}_backtest_ema_ratio.csv"

    if not os.path.exists(p1):
        raise FileNotFoundError(p1)
    if not os.path.exists(p2):
        raise FileNotFoundError(p2)

    bt1 = _load_bt(p1)
    bt2 = _load_bt(p2)

    rows = [
        _metrics(bt1, "Trend 1: MACD + Regime"),
        _metrics(bt2, "Trend 2: EMA Ratio"),
    ]
    out = pd.DataFrame(rows)

    out_path = f"{processed_dir}/{symbol}_{interval}_strategy_compare.csv"
    out.to_csv(out_path, index=False)

    logger.info(f"Saved strategy compare table: {out_path}")
    logger.info(f"\n{out.to_string(index=False)}")


if __name__ == "__main__":
    main()
