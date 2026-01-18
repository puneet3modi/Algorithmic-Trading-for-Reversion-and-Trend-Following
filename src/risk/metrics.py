from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskConfig:
    bars_per_year: int = 365 * 24 * 4  # 15m bars
    var_alpha: float = 0.01            # 1% VaR
    es_alpha: float = 0.01             # 1% ES


def _safe_series(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    return x.replace([np.inf, -np.inf], np.nan).dropna()


def annualized_sharpe(returns: pd.Series, bars_per_year: int) -> float:
    r = _safe_series(returns)
    if len(r) < 2:
        return float("nan")
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd <= 0:
        return float("nan")
    return float(np.sqrt(bars_per_year) * mu / sd)


def max_drawdown(equity: pd.Series) -> float:
    eq = _safe_series(equity)
    if len(eq) < 2:
        return float("nan")
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())


def turnover_sum(turnover: pd.Series) -> float:
    t = _safe_series(turnover)
    return float(t.sum()) if len(t) else 0.0


def pct_time_in_market(pos_exec: pd.Series) -> float:
    p = _safe_series(pos_exec)
    if len(p) == 0:
        return float("nan")
    return float((p.abs() > 0).mean())


def var_es(returns: pd.Series, alpha: float) -> tuple[float, float]:
    r = _safe_series(returns)
    if len(r) < 5:
        return float("nan"), float("nan")
    q = float(np.quantile(r, alpha))
    tail = r[r <= q]
    es = float(tail.mean()) if len(tail) else float("nan")
    return q, es


def realized_vol(returns: pd.Series, bars_per_year: int) -> float:
    r = _safe_series(returns)
    if len(r) < 2:
        return float("nan")
    return float(np.sqrt(bars_per_year) * r.std(ddof=1))


def equity_curve_stats(bt: pd.DataFrame, cfg: RiskConfig) -> dict:
    # Expect columns equity_gross/equity_net, strat_ret_gross/strat_ret_net, turnover, pos_exec
    out = {}

    for side in ["gross", "net"]:
        rcol = f"strat_ret_{side}"
        ecol = f"equity_{side}"
        if rcol in bt.columns:
            out[f"sharpe_{side}"] = annualized_sharpe(bt[rcol], cfg.bars_per_year)
            out[f"vol_{side}"] = realized_vol(bt[rcol], cfg.bars_per_year)
            v, es = var_es(bt[rcol], cfg.var_alpha)
            out[f"var_{int(cfg.var_alpha*100)}p_{side}"] = v
            out[f"es_{int(cfg.es_alpha*100)}p_{side}"] = es
        else:
            out[f"sharpe_{side}"] = float("nan")
            out[f"vol_{side}"] = float("nan")
            out[f"var_{int(cfg.var_alpha*100)}p_{side}"] = float("nan")
            out[f"es_{int(cfg.es_alpha*100)}p_{side}"] = float("nan")

        if ecol in bt.columns:
            out[f"max_dd_{side}"] = max_drawdown(bt[ecol])
            out[f"final_equity_{side}"] = float(_safe_series(bt[ecol]).iloc[-1]) if len(_safe_series(bt[ecol])) else float("nan")
        else:
            out[f"max_dd_{side}"] = float("nan")
            out[f"final_equity_{side}"] = float("nan")

    if "turnover" in bt.columns:
        out["total_turnover"] = turnover_sum(bt["turnover"])
    else:
        out["total_turnover"] = float("nan")

    if "pos_exec" in bt.columns:
        out["pct_time_in_market"] = pct_time_in_market(bt["pos_exec"])
    else:
        out["pct_time_in_market"] = float("nan")

    return out
