#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY_FILES: Dict[str, str] = {
    "Trend 1: MACD + Regime": "data/processed/BTCUSDT_15m_backtest_macd.csv",
    "Trend 2: EMA Ratio": "data/processed/BTCUSDT_15m_backtest_ema_ratio.csv",
    "Reversion: VWAP (LOCKED)": "data/processed/BTCUSDT_15m_backtest_reversion_vwap.csv",
    "ML: Classifier": "data/processed/BTCUSDT_15m_backtest_classifier.csv",
}

# 15-minute bars, ~365*24*4 = 35040 bars/year (crypto trades 24/7)
BARS_PER_YEAR = 365 * 24 * 4
ROLL_VOL_WINDOW = 4 * 24 * 7  # 1 week of 15m bars


def _read_backtest(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Find a time column
    time_cols = [c for c in df.columns if c.lower() in ("ts", "timestamp", "time", "datetime", "open_time", "date")]
    if time_cols:
        tcol = time_cols[0]
        df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
        df = df.sort_values(tcol).set_index(tcol)
    else:
        # fallback: try first column if it looks like datetime
        first = df.columns[0]
        dt = pd.to_datetime(df[first], utc=True, errors="coerce")
        if dt.notna().mean() > 0.8:
            df[first] = dt
            df = df.sort_values(first).set_index(first)

    return df


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    # try case-insensitive match
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _get_equity_series(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    eq_g = _pick_col(df, ["equity_gross", "equityGross", "equity", "equity_curve_gross", "equity_curve"])
    eq_n = _pick_col(df, ["equity_net", "equityNet", "equity_curve_net"])

    # If only returns exist, synthesize equity
    r_g = _pick_col(df, ["strategy_ret_gross", "ret_gross", "gross_ret", "strat_ret_gross"])
    r_n = _pick_col(df, ["strategy_ret_net", "ret_net", "net_ret", "strat_ret_net", "strategy_ret"])

    if eq_g is None and r_g is not None:
        eq_g_ser = (1.0 + df[r_g].fillna(0.0)).cumprod()
    elif eq_g is not None:
        eq_g_ser = df[eq_g].astype(float)
    else:
        eq_g_ser = pd.Series(index=df.index, data=np.nan)

    if eq_n is None and r_n is not None:
        eq_n_ser = (1.0 + df[r_n].fillna(0.0)).cumprod()
    elif eq_n is not None:
        eq_n_ser = df[eq_n].astype(float)
    else:
        eq_n_ser = pd.Series(index=df.index, data=np.nan)

    return eq_g_ser, eq_n_ser


def _get_net_returns(df: pd.DataFrame) -> pd.Series:
    r_n = _pick_col(df, ["strategy_ret_net", "ret_net", "net_ret", "strat_ret_net", "strategy_ret"])
    if r_n is None:
        # fallback: infer from equity_net
        _, eq_n = _get_equity_series(df)
        if eq_n.notna().any():
            return eq_n.pct_change().fillna(0.0)
        return pd.Series(index=df.index, data=np.nan)
    return df[r_n].astype(float).fillna(0.0)


def _drawdown(equity: pd.Series) -> pd.Series:
    eq = equity.dropna()
    if eq.empty:
        return equity
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return dd.reindex(equity.index)


def plot_equity_curves(all_data: Dict[str, pd.DataFrame]) -> None:
    plt.figure()
    for name, df in all_data.items():
        eq_g, eq_n = _get_equity_series(df)
        # plot net if available, else gross
        if eq_n.notna().any():
            plt.plot(eq_n.index, eq_n.values, label=f"{name} (net)")
        elif eq_g.notna().any():
            plt.plot(eq_g.index, eq_g.values, label=f"{name} (gross)")
    plt.title("Equity Curves (Net preferred)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "equity_curves.png", dpi=200)
    plt.close()


def plot_drawdowns(all_data: Dict[str, pd.DataFrame]) -> None:
    plt.figure()
    for name, df in all_data.items():
        _, eq_n = _get_equity_series(df)
        if eq_n.notna().any():
            dd = _drawdown(eq_n)
            plt.plot(dd.index, dd.values, label=name)
    plt.title("Drawdowns (Net)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "drawdowns.png", dpi=200)
    plt.close()


def plot_return_hist(all_data: Dict[str, pd.DataFrame]) -> None:
    plt.figure()
    for name, df in all_data.items():
        r = _get_net_returns(df)
        r = r.replace([np.inf, -np.inf], np.nan).dropna()
        if not r.empty:
            plt.hist(r.values, bins=60, alpha=0.35, label=name)
    plt.title("Distribution of Net Returns (15m bars)")
    plt.xlabel("Return per bar")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "returns_hist_net.png", dpi=200)
    plt.close()


def plot_rolling_vol(all_data: Dict[str, pd.DataFrame]) -> None:
    plt.figure()
    for name, df in all_data.items():
        r = _get_net_returns(df)
        r = r.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        vol = r.rolling(ROLL_VOL_WINDOW).std() * np.sqrt(BARS_PER_YEAR)
        if vol.notna().any():
            plt.plot(vol.index, vol.values, label=name)
    plt.title(f"Rolling Annualized Volatility (window={ROLL_VOL_WINDOW} bars)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Annualized vol")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rolling_vol_net.png", dpi=200)
    plt.close()

RISK_DASHBOARD_PATH = "data/processed/BTCUSDT_15m_risk_dashboard.csv"


def _try_read_risk_dashboard() -> Optional[pd.DataFrame]:
    if not os.path.exists(RISK_DASHBOARD_PATH):
        return None
    df = pd.read_csv(RISK_DASHBOARD_PATH)
    # normalize common column names
    if "strategy" not in df.columns:
        # try case-insensitive
        lower_map = {c.lower(): c for c in df.columns}
        if "strategy" in lower_map:
            df = df.rename(columns={lower_map["strategy"]: "strategy"})
    return df


def _infer_turnover_from_positions(df: pd.DataFrame) -> float:
    """
    Fallback turnover proxy if risk dashboard not available.
    Uses sum(|Δposition|) where position ∈ {-1,0,1} or {0,1}.
    """
    pos_col = _pick_col(df, ["position", "pos", "strategy_pos", "target_position", "signal_position"])
    if pos_col is None:
        # some backtests store positions with strategy-specific column names
        for c in df.columns:
            if "position" in c.lower():
                pos_col = c
                break
    if pos_col is None:
        return float("nan")

    pos = pd.to_numeric(df[pos_col], errors="coerce").fillna(0.0)
    dpos = pos.diff().abs().fillna(0.0)
    return float(dpos.sum())


def plot_turnover(all_data: Dict[str, pd.DataFrame]) -> None:
    """
    Preferred source: risk dashboard total_turnover (already computed in your pipeline).
    Fallback: inferred turnover proxy from position changes.
    """
    rd = _try_read_risk_dashboard()

    labels: List[str] = []
    values: List[float] = []

    if rd is not None and "total_turnover" in rd.columns and "strategy" in rd.columns:
        # Use dashboard directly
        # Map dashboard strategy rows to our plotting names (they should match)
        for name in all_data.keys():
            hit = rd[rd["strategy"] == name]
            if not hit.empty:
                v = float(hit["total_turnover"].iloc[0])
            else:
                v = _infer_turnover_from_positions(all_data[name])
            labels.append(name)
            values.append(v)
    else:
        # Infer from positions
        for name, df in all_data.items():
            labels.append(name)
            values.append(_infer_turnover_from_positions(df))

    # Plot as bar chart
    plt.figure()
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.title("Turnover (Total) — Dashboard or Position-Change Proxy")
    plt.xlabel("Strategy")
    plt.ylabel("Total turnover (units vary by definition)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "turnover.png", dpi=200)
    plt.close()

def main() -> None:
    all_data: Dict[str, pd.DataFrame] = {}
    missing = []
    for strat, path in STRATEGY_FILES.items():
        if not os.path.exists(path):
            missing.append(path)
            continue
        all_data[strat] = _read_backtest(path)

    if missing:
        print("WARNING: Missing backtest CSVs:")
        for m in missing:
            print(" -", m)
        print("Plots will be generated for available strategies only.\n")

    if not all_data:
        raise SystemExit("No backtest CSVs found. Nothing to plot.")

    plot_equity_curves(all_data)
    plot_drawdowns(all_data)
    plot_return_hist(all_data)
    plot_rolling_vol(all_data)
    plot_turnover(all_data)

    print("Saved figures to:", FIG_DIR.resolve())
    for f in ["equity_curves.png", "drawdowns.png", "returns_hist_net.png", "rolling_vol_net.png"]:
        fp = FIG_DIR / f
        if fp.exists():
            print(" -", fp)


if __name__ == "__main__":
    main()
    