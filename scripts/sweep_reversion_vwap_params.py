from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import yaml

sys.path.append(os.path.abspath("."))

from src.common.logging import setup_logger
from src.indicators.ewma_vol import EWMAVolParams, ewma_vol
from src.indicators.vwap import rolling_vwap
from src.strategies.vwap_reversion import VWAPReversionParams, generate_positions_vwap_reversion
from src.backtest.simple_backtest import BacktestParams, run_backtest, annualized_sharpe, max_drawdown


def main() -> None:
    logger = setup_logger("sweep_reversion_vwap_params")

    with open("config/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    symbol = cfg["symbol"]
    interval = cfg["timebars"]["interval"]
    processed_dir = cfg["paths"]["processed_dir"]

    base_path = f"{processed_dir}/{symbol}_{interval}_klines_processed.csv"
    ema_path = f"{processed_dir}/{symbol}_{interval}_ema_ratio_positions.csv"

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Missing: {base_path}")
    if not os.path.exists(ema_path):
        raise FileNotFoundError(f"Missing: {ema_path}")

    base = pd.read_csv(base_path)
    base["open_time"] = pd.to_datetime(base["open_time"], utc=True)
    base = base.sort_values("open_time").set_index("open_time")

    ema = pd.read_csv(ema_path)
    ema["open_time"] = pd.to_datetime(ema["open_time"], utc=True)
    ema = ema.sort_values("open_time").set_index("open_time")

    df = base.join(ema[["ema_ratio"]], how="left")
    if df["ema_ratio"].isna().any():
        df["ema_ratio"] = df["ema_ratio"].ffill()

    close = df["close"].astype(float)
    volu = df["volume"].astype(float)

    logret = np.log(close).diff().rename("logret")
    df["ewma_vol"] = ewma_vol(logret, EWMAVolParams(lam=0.94, annualize=False))

    abs_tr = df["ema_ratio"].abs()
    q_levels = [0.2, 0.3, 0.4, 0.5]
    gates = {q: float(abs_tr.quantile(q)) for q in q_levels}
    logger.info(f"Trend gates (quantiles of |ema_ratio|): {gates}")

    vwap_windows = [96, 288, 480]  # 1d, 3d, 5d
    k_entries = [1.25, 1.5, 1.75, 2.0, 2.25]
    holds = [8, 16, 32]

    k_exit = 0.5
    stop_k = 4.0
    cooldown = 1

    cost = 0.0002
    exec_lag = 1
    bars_per_year = 365 * 24 * 4

    rows = []
    for vw in vwap_windows:
        vwap = rolling_vwap(close, volu, window=vw)
        dist = np.log(close / vwap).rename("dist")

        tmp_base = df.copy()
        tmp_base["vwap"] = vwap
        tmp_base["dist"] = dist

        for q in q_levels:
            for k in k_entries:
                for mh in holds:
                    params = VWAPReversionParams(
                        vwap_window=vw,
                        k_entry=k,
                        k_exit=k_exit,
                        max_hold_bars=mh,
                        cooldown_bars=cooldown,
                        trend_gate=gates[q],
                        stop_k=stop_k,
                        mode="long_short",
                    )

                    pos = generate_positions_vwap_reversion(tmp_base[["dist", "ewma_vol", "ema_ratio"]], params)
                    tmp = tmp_base.copy()
                    tmp["position"] = pos

                    bt = run_backtest(tmp, BacktestParams(cost_per_turnover=cost, execution_lag=exec_lag))

                    rows.append(
                        {
                            "vwap_window": vw,
                            "trend_gate_quantile": q,
                            "trend_gate_value": gates[q],
                            "k_entry": k,
                            "max_hold_bars": mh,
                            "pct_time_in_mkt": float((bt["pos_exec"].abs() > 0).mean()),
                            "turnover": float(bt["turnover"].sum()),
                            "sharpe_gross": annualized_sharpe(bt["strat_ret_gross"], bars_per_year),
                            "sharpe_net": annualized_sharpe(bt["strat_ret_net"], bars_per_year),
                            "mdd_net": max_drawdown(bt["equity_net"]),
                            "final_equity_net": float(bt["equity_net"].iloc[-1]),
                        }
                    )

    res = pd.DataFrame(rows).sort_values(["sharpe_net", "final_equity_net"], ascending=False)
    out_path = f"{processed_dir}/{symbol}_{interval}_reversion_vwap_param_sweep.csv"
    res.to_csv(out_path, index=False)

    logger.info(f"Saved VWAP reversion sweep: {out_path}")
    logger.info("Top 20 configs:")
    logger.info("\n" + res.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
