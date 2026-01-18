from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
import argparse
import pandas as pd
import yaml

from src.common.logging import setup_logger
from src.risk.metrics import RiskConfig, equity_curve_stats


def load_bt(path: str) -> pd.DataFrame:
    bt = pd.read_csv(path)
    bt["open_time"] = pd.to_datetime(bt["open_time"], utc=True)
    bt = bt.sort_values("open_time").set_index("open_time")
    return bt


def main() -> None:
    logger = setup_logger("report_risk_dashboard")

    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="Report all known strategy backtests")
    ap.add_argument("--path", type=str, default="", help="Optional: single backtest csv path")
    args = ap.parse_args()

    with open("config/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    symbol = cfg["symbol"]
    interval = cfg["timebars"]["interval"]
    processed_dir = cfg["paths"]["processed_dir"]

    cfg_risk = RiskConfig()

    # Strategy files (extend as needed)
    candidates = {
        "Trend 1: MACD + Regime": f"{processed_dir}/{symbol}_{interval}_backtest_macd.csv",
        "Trend 2: EMA Ratio": f"{processed_dir}/{symbol}_{interval}_backtest_ema_ratio.csv",
        "Reversion: VWAP (LOCKED)": f"{processed_dir}/{symbol}_{interval}_backtest_reversion_vwap.csv",
        "ML: Classifier": f"{processed_dir}/{symbol}_{interval}_backtest_classifier.csv",
    }

    paths = []
    if args.path:
        paths = [("Custom", args.path)]
    elif args.all:
        for name, p in candidates.items():
            if os.path.exists(p):
                paths.append((name, p))
            else:
                logger.warning(f"Missing backtest file (skipping): {p}")
    else:
        # Default: report the compare table strategies (not ML)
        for name in ["Trend 1: MACD + Regime", "Trend 2: EMA Ratio", "Reversion: VWAP (LOCKED)"]:
            p = candidates[name]
            if os.path.exists(p):
                paths.append((name, p))

    rows = []
    for name, p in paths:
        bt = load_bt(p)
        stats = equity_curve_stats(bt, cfg_risk)
        stats["strategy"] = name
        stats["path"] = p
        stats["period_start_utc"] = str(bt.index.min())
        stats["period_end_utc"] = str(bt.index.max())
        rows.append(stats)

    out = pd.DataFrame(rows)
    out_path = f"{processed_dir}/{symbol}_{interval}_risk_dashboard.csv"
    out.to_csv(out_path, index=False)

    logger.info(f"Saved risk dashboard: {out_path}")
    logger.info("\n" + out.to_string(index=False))


if __name__ == "__main__":
    main()
    