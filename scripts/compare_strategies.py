from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import yaml

sys.path.append(os.path.abspath("."))

from src.common.logging import setup_logger
from src.backtest.simple_backtest import max_drawdown


def _load_bt(path: str) -> pd.DataFrame:
    bt = pd.read_csv(path)
    bt["open_time"] = pd.to_datetime(bt["open_time"], utc=True)
    bt = bt.sort_values("open_time").set_index("open_time")
    return bt


def _equity_to_rets(equity: pd.Series) -> pd.Series:
    # equity is multiplicative (starts ~1.0), so simple returns are fine
    r = equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return r


def _annualized_sharpe_from_equity(equity: pd.Series, bars_per_year: int) -> float:
    r = _equity_to_rets(equity)
    sd = float(r.std(ddof=0))
    if sd == 0.0:
        return 0.0
    return float(np.sqrt(bars_per_year) * r.mean() / sd)


def _metrics(bt: pd.DataFrame, name: str) -> dict:
    bars_per_year = 365 * 24 * 4  # 15m bars

    # Sharpe from equity curves (works even if strat_ret_* columns are not saved)
    sharpe_g = _annualized_sharpe_from_equity(bt["equity_gross"], bars_per_year)
    sharpe_n = _annualized_sharpe_from_equity(bt["equity_net"], bars_per_year)

    return {
        "strategy": name,
        "final_equity_gross": float(bt["equity_gross"].iloc[-1]),
        "final_equity_net": float(bt["equity_net"].iloc[-1]),
        "sharpe_gross": sharpe_g,
        "sharpe_net": sharpe_n,
        "max_dd_gross": max_drawdown(bt["equity_gross"]),
        "max_dd_net": max_drawdown(bt["equity_net"]),
        "total_turnover": float(bt["turnover"].sum()) if "turnover" in bt.columns else float("nan"),
        "pct_time_in_market": float((bt["pos_exec"].abs() > 0).mean()) if "pos_exec" in bt.columns else float("nan"),
    }


def main() -> None:
    logger = setup_logger("compare_strategies")

    with open("config/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    symbol = cfg["symbol"]
    interval = cfg["timebars"]["interval"]
    processed_dir = cfg["paths"]["processed_dir"]

    p_macd = f"{processed_dir}/{symbol}_{interval}_backtest_macd.csv"
    p_ema_ratio = f"{processed_dir}/{symbol}_{interval}_backtest_ema_ratio.csv"
    p_reversion_vwap = f"{processed_dir}/{symbol}_{interval}_backtest_reversion_vwap.csv"
    p_ml = f"{processed_dir}/{symbol}_{interval}_backtest_classifier.csv"

    paths = [
        (p_macd, "Trend 1: MACD + Regime"),
        (p_ema_ratio, "Trend 2: EMA Ratio"),
        (p_reversion_vwap, "Reversion: VWAP (LOCKED)"),
        (p_ml, "Alt: ML Classifier (Stacking)"),
    ]

    for p, _name in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    rows = []
    for p, name in paths:
        bt = _load_bt(p)
        rows.append(_metrics(bt, name))

    out = pd.DataFrame(rows)

    out_path = f"{processed_dir}/{symbol}_{interval}_strategy_compare.csv"
    out.to_csv(out_path, index=False)

    logger.info(f"Saved strategy compare table: {out_path}")
    logger.info(f"\n{out.to_string(index=False)}")


if __name__ == "__main__":
    main()
