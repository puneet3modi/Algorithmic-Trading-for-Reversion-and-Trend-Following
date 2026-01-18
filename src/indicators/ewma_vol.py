from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EWMAVolParams:
    lam: float = 0.94
    annualize: bool = False
    bars_per_year: int = 365 * 24 * 4
    eps: float = 1e-12


def ewma_vol(logret: pd.Series, params: EWMAVolParams) -> pd.Series:
    lam = float(params.lam)
    if not (0.0 < lam < 1.0):
        raise ValueError("lam must be in (0,1)")

    r = logret.astype(float).to_numpy()
    n = len(r)

    var = np.full(n, np.nan, dtype=float)
    # initialize with sample variance of first chunk
    init_idx = min(100, n)
    init_var = np.nanvar(r[:init_idx]) if init_idx > 5 else 0.0

    v = init_var
    for i in range(n):
        if np.isfinite(r[i]):
            v = lam * v + (1.0 - lam) * (r[i] ** 2)
        var[i] = v

    vol = np.sqrt(np.maximum(var, 0.0))
    if params.annualize:
        vol = vol * np.sqrt(params.bars_per_year)

    return pd.Series(vol, index=logret.index, name="ewma_vol")
