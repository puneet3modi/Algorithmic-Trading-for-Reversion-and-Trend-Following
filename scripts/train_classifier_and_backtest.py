from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import yaml

sys.path.append(os.path.abspath("."))

from src.common.logging import setup_logger
from src.ml.features import add_basic_ml_features
from src.ml.stacking import StackingParams, fit_predict_stacking
from src.backtest.simple_backtest import BacktestParams, run_backtest, annualized_sharpe, max_drawdown


def make_labels(logret_fwd: pd.Series, tau: float) -> pd.Series:
    y = pd.Series(index=logret_fwd.index, dtype="float")
    y[logret_fwd >  tau] = 1
    y[logret_fwd < -tau] = 0
    return y

def probs_to_position_stateful(
    p: pd.Series,
    p_long: float = 0.55,
    p_short: float = 0.45,
    exit_to_flat: float = 0.50,
    min_hold_bars: int = 4,
    cooldown_bars: int = 2,
) -> pd.Series:
    """
    Converts predicted probabilities into positions using:
    - Entry band: long if p>=p_long, short if p<=p_short
    - Exit rule: after min-hold, exit to flat if p crosses exit_to_flat
    - Cooldown: forced flat for cooldown_bars after an exit
    This reduces turnover vs naive thresholding.
    """
    idx = p.index
    pos = np.zeros(len(p), dtype=int)

    current = 0   # -1, 0, +1
    hold = 0      # bars since entry
    cool = 0      # cooldown remaining

    pv = p.values

    for t in range(len(pv)):
        if cool > 0:
            current = 0
            cool -= 1
            pos[t] = current
            continue

        if current != 0:
            hold += 1
            if hold >= min_hold_bars:
                # exit to flat if confidence weakens past exit threshold
                if current == 1 and pv[t] <= exit_to_flat:
                    current = 0
                    hold = 0
                    cool = cooldown_bars
                elif current == -1 and pv[t] >= exit_to_flat:
                    current = 0
                    hold = 0
                    cool = cooldown_bars
            pos[t] = current
            continue

        # currently flat: can enter if confident
        if pv[t] >= p_long:
            current = 1
            hold = 0
        elif pv[t] <= p_short:
            current = -1
            hold = 0
        else:
            current = 0

        pos[t] = current

    return pd.Series(pos, index=idx, dtype=int)


def main() -> None:
    logger = setup_logger("train_classifier_and_backtest")

    with open("config/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    symbol = cfg["symbol"]
    interval = cfg["timebars"]["interval"]
    processed_dir = cfg["paths"]["processed_dir"]

    # Use the richest dataset we have: MACD norms + EMA ratio + (optionally) VWAP dist, ewma_vol if present.
    base_path = f"{processed_dir}/{symbol}_{interval}_klines_processed.csv"
    macd_path = f"{processed_dir}/{symbol}_{interval}_with_macd.csv"
    ema_path = f"{processed_dir}/{symbol}_{interval}_ema_ratio_positions.csv"

    df = pd.read_csv(base_path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").set_index("open_time")

    macd = pd.read_csv(macd_path)
    macd["open_time"] = pd.to_datetime(macd["open_time"], utc=True)
    macd = macd.sort_values("open_time").set_index("open_time")

    ema = pd.read_csv(ema_path)
    ema["open_time"] = pd.to_datetime(ema["open_time"], utc=True)
    ema = ema.sort_values("open_time").set_index("open_time")

    df = df.join(macd.drop(columns=["open", "high", "low", "close", "volume"], errors="ignore"), how="left")
    df = df.join(ema[["ema_ratio"]], how="left")
    df["ema_ratio"] = df["ema_ratio"].ffill()

    # Add ML features
    df = add_basic_ml_features(df)

    # Label: next-bar log return sign (optionally with tau)
    close = df["close"].astype(float)
    df["logret"] = np.log(close).diff()
    h = 4  # 4*15m = 1 hour
    df["logret_fwd"] = df["logret"].rolling(h).sum().shift(-h)


    # tau choice: small noise filter (report this!)
    tau = 0.0005
    df["y"] = make_labels(df["logret_fwd"], tau)

    feature_cols = [
        # core
        "hist_norm", "macd_norm", "signal_norm", "ema_ratio",
        # dynamics
        "logret", "ret_mean_32", "ret_std_32", "vol_z_96", "mom_32", "mom_96",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    data = df.dropna(subset=feature_cols + ["y", "logret_fwd"]).copy()

    # Walk-forward split: 70% train, 30% test
    n = len(data)
    split = int(0.7 * n)
    train = data.iloc[:split]
    test = data.iloc[split:]

    X_tr = train[feature_cols].to_numpy(dtype=float)
    y_tr = train["y"].to_numpy(dtype=int)
    X_te = test[feature_cols].to_numpy(dtype=float)
    y_te = test["y"].to_numpy(dtype=int)

    # Standardize features based on train only
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0) + 1e-12
    X_trs = (X_tr - mu) / sd
    X_tes = (X_te - mu) / sd

    p_test, info = fit_predict_stacking(X_trs, y_tr, X_tes, StackingParams())

    logger.info(f"Train AUCs: {info}")

    p = pd.Series(p_test, index=test.index, name="p_up")

    # IC: corr(p, forward return)
    ic = float(np.corrcoef(p.values, test["logret_fwd"].values)[0, 1])
    logger.info(f"Information Coefficient (IC) on test: {ic:.4f}")

    # Convert to positions + backtest on test period only
    pos = probs_to_position_stateful(
        p,
        p_long=0.56,
        p_short=0.44,
        exit_to_flat=0.50,
        min_hold_bars=8,
        cooldown_bars=2,
    ).rename("position")


    bt_df = test.copy()
    bt_df["position"] = pos

    # Costs and execution assumptions consistent with your other work
    cost = 0.0002   # 2 bps per unit turnover
    exec_lag = 1
    bars_per_year = 365 * 24 * 4

    bt = run_backtest(bt_df, BacktestParams(cost_per_turnover=cost, execution_lag=exec_lag))

    sharpe_g = annualized_sharpe(bt["strat_ret_gross"], bars_per_year)
    sharpe_n = annualized_sharpe(bt["strat_ret_net"], bars_per_year)
    mdd_n = max_drawdown(bt["equity_net"])
    turnover = float(bt["turnover"].sum())
    pct_mkt = float((bt["pos_exec"].abs() > 0).mean())

    logger.info(f"Classifier strategy Sharpe gross: {sharpe_g:.3f} | net: {sharpe_n:.3f}")
    logger.info(f"Max DD net: {mdd_n:.3%} | turnover: {turnover:.0f} | time in market: {pct_mkt:.3f}")

    out_pred = f"{processed_dir}/{symbol}_{interval}_classifier_preds.csv"
    out_bt = f"{processed_dir}/{symbol}_{interval}_backtest_classifier.csv"

    outp = pd.DataFrame({"p_up": p, "position": pos, "logret_fwd": test["logret_fwd"]})
    outp.to_csv(out_pred)

    bt_out = bt[["equity_gross", "equity_net", "pos_exec", "turnover"]].copy()
    bt_out.to_csv(out_bt)

    logger.info(f"Saved predictions: {out_pred}")
    logger.info(f"Saved backtest: {out_bt}")


if __name__ == "__main__":
    main()
