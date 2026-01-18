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
from src.backtest.simple_backtest import BacktestParams, run_backtest, annualized_sharpe, max_drawdown


def main() -> None:
    logger = setup_logger("sweep_reversion_params")

    with open("config/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    symbol = cfg["symbol"]
    interval = cfg["timebars"]["interval"]
    processed_dir = cfg["paths"]["processed_dir"]

    in_path = f"{processed_dir}/{symbol}_{interval}_ema_ratio_positions.csv"
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Missing: {in_path}")

    df = pd.read_csv(in_path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").set_index("open_time")

    close = df["close"].astype(float)
    logret = np.log(close).diff().rename("logret")
    vol = ewma_vol(logret, EWMAVolParams(lam=0.94, annualize=False))
    df["shock"] = (logret / (vol + 1e-12)).rename("shock")

    # Precompute quantiles for trend gating
    abs_tr = df["ema_ratio"].abs()
    q_levels = [0.2, 0.3, 0.4, 0.5]
    gates = {q: float(abs_tr.quantile(q)) for q in q_levels}
    logger.info(f"Trend gate quantiles (abs ema_ratio): {gates}")

    bars_per_year = 365 * 24 * 4

    k_entries = [1.0, 1.25, 1.5, 1.75, 2.0]
    k_exit = 0.5
    holds = [8, 16, 32]

    cost = 0.0002  # 2 bps / turnover
    exec_lag = 1

    rows = []
    for q in q_levels:
        for k in k_entries:
            for mh in holds:
                sparams = ShockReversionParams(
                    k_entry=k,
                    k_exit=k_exit,
                    trend_gate=gates[q],
                    max_hold_bars=mh,
                    cooldown_bars=1,
                    mode="long_short",
                )

                pos = generate_positions_shock_reversion(df[["shock", "ema_ratio"]], sparams)
                tmp = df.copy()
                tmp["position"] = pos

                bt = run_backtest(tmp, BacktestParams(cost_per_turnover=cost, execution_lag=exec_lag))

                rows.append(
                    {
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

    out_path = f"{processed_dir}/{symbol}_{interval}_reversion_param_sweep.csv"
    res.to_csv(out_path, index=False)

    logger.info(f"Saved param sweep: {out_path}")
    logger.info("Top 15 configs:")
    logger.info("\n" + res.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
