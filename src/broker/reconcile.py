from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class ReconcileResult:
    symbol: str
    desired_position: int         # -1/0/+1 from strategy (for logging)
    shadow_position: int          # 0/1 inferred from account balances (spot)
    base_asset: str
    quote_asset: str
    base_free: float
    base_locked: float
    quote_free: float
    quote_locked: float
    base_mark_px: float
    base_value_quote: float       # base_total * px
    reason: str
    ok: bool


def split_symbol_spot(symbol: str) -> tuple[str, str]:
    """
    For this project we assume USDT quote for Binance spot symbols.
    BTCUSDT -> (BTC, USDT)
    """
    if symbol.endswith("USDT"):
        return symbol[:-4], "USDT"
    return symbol[:-3], symbol[-3:]


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def infer_shadow_position_spot(
    symbol: str,
    account: Dict[str, Any],
    last_px: float,
    desired_position: int,
    notional_usdt: float,
    min_notional_usdt: float = 5.0,
) -> ReconcileResult:
    """
    Infers a coarse spot shadow position in {0,1} from balances.

    Spot cannot be structurally short (without margin/futures), so:
    shadow_position = 1 if base_value_quote >= threshold else 0

    threshold uses max(min_notional_usdt, 0.5 * notional_usdt).
    """
    base, quote = split_symbol_spot(symbol)

    balances = account.get("balances", [])
    bal_map = {b.get("asset"): b for b in balances}

    b = bal_map.get(base, {})
    q = bal_map.get(quote, {})

    base_free = _to_float(b.get("free"))
    base_locked = _to_float(b.get("locked"))
    quote_free = _to_float(q.get("free"))
    quote_locked = _to_float(q.get("locked"))

    base_total = base_free + base_locked
    quote_total = quote_free + quote_locked

    px = float(last_px)
    base_value = base_total * px

    threshold = max(float(min_notional_usdt), 0.5 * float(notional_usdt))
    shadow = 1 if base_value >= threshold else 0

    reason = (
        f"base_total={base_total:.8f} {base} (~{base_value:.2f} {quote}) "
        f"quote_total={quote_total:.2f} {quote} threshold={threshold:.2f}"
    )

    return ReconcileResult(
        symbol=symbol,
        desired_position=int(desired_position),
        shadow_position=int(shadow),
        base_asset=base,
        quote_asset=quote,
        base_free=base_free,
        base_locked=base_locked,
        quote_free=quote_free,
        quote_locked=quote_locked,
        base_mark_px=px,
        base_value_quote=base_value,
        reason=reason,
        ok=True,
    )


def reconcile_desired_vs_shadow_spot(desired_position: int, shadow_position: int) -> int:
    """
    For spot:
    desired +1 -> target 1
    desired  0 -> target 0
    desired -1 -> interpret as flat (target 0)

    Returns target_position in {0,1}.
    """
    return 1 if desired_position > 0 else 0


def should_trade(
    target_position: int,
    shadow_position: int,
    open_orders_count: int,
) -> tuple[bool, str]:
    """
    Trade gate:
    - don't stack orders
    - don't trade if already at target
    """
    if open_orders_count > 0:
        return False, f"skip: open_orders_count={open_orders_count}"
    if target_position == shadow_position:
        return False, f"skip: already at target_position={target_position}"
    return True, "ok"
