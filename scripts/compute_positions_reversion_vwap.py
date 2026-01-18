from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import yaml

sys.path.append(os.path.abspath("."))

from src.common.logging import setup_logger

VWAP_WINDOW = 480                 # bars
TREND_GATE_QUANTILE = 0.20        # quantile of |ema_ratio|
TREND_GATE_VALUE = 0.0013250666   # locked from sweep (printed)
K_ENTRY = 2.25                    # entry threshold in sigma units of dist
MAX_HOLD_BARS = 32                # time stop


def rolling_vwap(close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    pv = close * volume
    pv_sum = pv.rolling(window=window, min_periods=window).sum()
    v_sum = volume.rolling(window=window, min_periods=window).sum()
    return pv_sum / (v_sum + 1e-12)


def main() -> None:
    logger = setup_logger("compute_positions_reversion_vwap")

    with open("config/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    symbol = cfg["symbol"]
    interval = cfg["timebars"]["interval"]
    processed_dir = cfg["paths"]["processed_dir"]

    base_path = f"{processed_dir}/{symbol}_{interval}_klines_processed.csv"
    ema_path = f"{processed_dir}/{symbol}_{interval}_ema_ratio_positions.csv"
    out_path = f"{processed_dir}/{symbol}_{interval}_reversion_vwap_positions.csv"

    df = pd.read_csv(base_path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").set_index("open_time")

    for col in ["close", "volume"]:
        if col not in df.columns:
            raise KeyError(f"Required column missing in base dataset: {col}")

    close = df["close"].astype(float)
    vol = df["volume"].astype(float)

    # Trend gate: ema_ratio
    ema = pd.read_csv(ema_path)
    ema["open_time"] = pd.to_datetime(ema["open_time"], utc=True)
    ema = ema.sort_values("open_time").set_index("open_time")

    df = df.join(ema[["ema_ratio"]], how="left")
    missing = int(df["ema_ratio"].isna().sum())
    if missing > 0:
        logger.warning(f"ema_ratio has {missing} missing values after join; forward-filling.")
        df["ema_ratio"] = df["ema_ratio"].ffill()

    q = float(df["ema_ratio"].abs().quantile(TREND_GATE_QUANTILE))
    logger.info(f"Trend gates (check only): quantile={TREND_GATE_QUANTILE:.2f} -> {q:.6f}")
    logger.info(f"Trend gate LOCKED value: {TREND_GATE_VALUE:.10f}")

    # VWAP + distance signal
    df["vwap_480"] = rolling_vwap(close, vol, VWAP_WINDOW)
    df["dist"] = (close / df["vwap_480"]) - 1.0

    # Standardize dist by rolling sigma of dist (same window)
    df["dist_sigma"] = df["dist"].rolling(window=VWAP_WINDOW, min_periods=VWAP_WINDOW).std()
    df["z_dist"] = df["dist"] / (df["dist_sigma"] + 1e-12)

    # Entry logic (mean reversion)
    # Trade only when NOT trending: |ema_ratio| <= trend_gate_value
    regime_ok = df["ema_ratio"].abs() <= TREND_GATE_VALUE

    # Reversion: if price below VWAP too much -> long; above VWAP too much -> short
    long_entry = regime_ok & (df["z_dist"] <= -K_ENTRY)
    short_entry = regime_ok & (df["z_dist"] >= K_ENTRY)

    # Position with time stop (MAX_HOLD_BARS) and exit on crossing VWAP (dist sign flip)
    pos = pd.Series(0, index=df.index, dtype=int)

    current = 0
    hold = 0

    for i in range(len(df)):
        if np.isnan(df["vwap_480"].iat[i]) or np.isnan(df["dist_sigma"].iat[i]):
            pos.iat[i] = 0
            current = 0
            hold = 0
            continue

        if current == 0:
            if long_entry.iat[i]:
                current = 1
                hold = 0
            elif short_entry.iat[i]:
                current = -1
                hold = 0
        else:
            hold += 1

            # Exit conditions:
            # 1) time stop
            if hold >= MAX_HOLD_BARS:
                current = 0
                hold = 0
            else:
                # 2) mean reversion completed: dist crosses 0
                d = df["dist"].iat[i]
                if current == 1 and d >= 0:
                    current = 0
                    hold = 0
                elif current == -1 and d <= 0:
                    current = 0
                    hold = 0

        pos.iat[i] = current

    out = pd.DataFrame(
        {
            "close": close,
            "vwap_480": df["vwap_480"],
            "dist": df["dist"],
            "ema_ratio": df["ema_ratio"],
            "position_reversion": pos,
        },
        index=df.index,
    )

    out.to_csv(out_path)

    logger.info(f"Saved VWAP reversion positions: {out_path}")
    logger.info("Position counts:")
    logger.info(out["position_reversion"].value_counts().to_string())
    logger.info("Tail:")
    logger.info(out.tail().to_string())


if __name__ == "__main__":
    main()
