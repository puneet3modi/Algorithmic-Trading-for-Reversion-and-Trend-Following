from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestParams:
    # cost in decimal terms per 1 unit of notional traded, e.g. 0.0005 = 5 bps
    cost_per_turnover: float = 0.0000
    # execution lag in bars: 1 = trade at next bar
    execution_lag: int = 1
    # position column name
    position_col: str = "position"
    # price column name for returns
    price_col: str = "close"


def compute_bar_returns(price: pd.Series) -> pd.Series:
    price = price.astype(float)
    return price.pct_change().rename("ret")


def compute_turnover(position: pd.Series) -> pd.Series:
    """
    Turnover proxy for a single-asset strategy:
    turnover_t = |pos_t - pos_{t-1}|
    where pos in {-1,0,1}; thus flips cost 2, enter/exit cost 1.
    """
    pos = position.fillna(0).astype(float)
    return pos.diff().abs().fillna(0.0).rename("turnover")


def run_backtest(df: pd.DataFrame, params: BacktestParams) -> pd.DataFrame:
    """
    Inputs df must contain:
    - price_col (close)
    - position_col (desired position at time t)

    Output includes:
    ret, pos_exec, strat_ret_gross, turnover, costs, strat_ret_net, equity
    """
    out = df.copy()

    ret = compute_bar_returns(out[params.price_col])
    out["ret"] = ret

    pos = out[params.position_col].fillna(0).astype(float)

    # Apply execution lag: position used for returns at t is position decided at t-lag
    lag = int(params.execution_lag)
    if lag < 0:
        raise ValueError("execution_lag must be >= 0")
    pos_exec = pos.shift(lag).fillna(0.0).rename("pos_exec")
    out["pos_exec"] = pos_exec

    # Gross strategy return: pos_exec * asset return
    out["strat_ret_gross"] = (out["pos_exec"] * out["ret"]).fillna(0.0)

    # Turnover from executed position changes
    turnover = compute_turnover(out["pos_exec"])
    out["turnover"] = turnover

    # Costs applied proportional to turnover
    c = float(params.cost_per_turnover)
    out["costs"] = (c * out["turnover"]).fillna(0.0)

    out["strat_ret_net"] = out["strat_ret_gross"] - out["costs"]

    # Equity curve (start at 1.0)
    out["equity_net"] = (1.0 + out["strat_ret_net"]).cumprod()
    out["equity_gross"] = (1.0 + out["strat_ret_gross"]).cumprod()

    return out


def max_drawdown(equity: pd.Series) -> float:
    eq = equity.astype(float)
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())


def annualized_sharpe(returns: pd.Series, bars_per_year: float) -> float:
    r = returns.dropna().astype(float)
    if len(r) < 2:
        return float("nan")
    mu = r.mean()
    sig = r.std(ddof=1)
    if sig == 0:
        return float("nan")
    return float((mu / sig) * np.sqrt(bars_per_year))
