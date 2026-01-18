from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import yaml

sys.path.append(os.path.abspath("."))

from src.common.logging import setup_logger
from src.indicators.ewma_vol import EWMAVolParams, ewma_vol
from src.strategies.shock_reversion import ShockReversionParams, generate_positions_shock_reversion


def main() -> None:
    logger = setup_logger("compute_positions_reversion")

    with open("config/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    symbol = cfg["symbol"]
    interval = cfg["timebars"]["interval"]
    processed_dir = cfg["paths"]["processed_dir"]

    in_path = f"{processed_dir}/{symbol}_{interval}_ema_ratio_positions.csv"
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Missing: {in_path}. Run Strategy 2 positions first.")

    df = pd.read_csv(in_path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").set_index("open_time")

    # log returns
    close = df["close"].astype(float)
    logret = np.log(close).diff().rename("logret")

    vol = ewma_vol(logret, EWMAVolParams(lam=0.94, annualize=False))
    shock = (logret / (vol + 1e-12)).rename("shock")

    df["shock"] = shock
    df["ewma_vol"] = vol

    params = ShockReversionParams(
        k_entry=2.0,
        k_exit=0.5,
        trend_gate=0.0010,
        max_hold_bars=16,
        cooldown_bars=1,
        mode="long_short",
    )

    df["position_reversion"] = generate_positions_shock_reversion(df[["shock", "ema_ratio"]], params)

    out_path = f"{processed_dir}/{symbol}_{interval}_reversion_positions.csv"
    df[["close", "ema_ratio", "shock", "ewma_vol", "position_reversion"]].to_csv(out_path)

    logger.info(f"Saved reversion positions: {out_path}")
    logger.info(f"Position counts:\n{df['position_reversion'].value_counts(dropna=False)}")
    logger.info(f"Tail:\n{df[['close','ema_ratio','shock','position_reversion']].tail(5)}")


if __name__ == "__main__":
    main()
