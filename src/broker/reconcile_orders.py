from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set


@dataclass(frozen=True)
class OrderReconcileResult:
    ok: bool
    open_orders_count: int
    open_order_ids: Set[int]
    expected_open_order_ids: Set[int]
    missing_expected_open_orders: List[int]
    unexpected_open_orders: List[int]
    any_open_order_executed_qty_gt_0: bool
    recent_trades_count: int | None
    reason: str


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def reconcile_open_orders_and_trades(
    client,
    *,
    symbol: str,
    expected_open_order_ids: Set[int],
    check_trades: bool = True,
    trades_limit: int = 10,
) -> OrderReconcileResult:
    """
    Broker-truth reconciliation:

    - open_orders(): what the broker currently has working
    - expected_open_order_ids: what we think should be open (based on submissions)

    Flags:
    - missing expected open orders (order disappeared)
    - unexpected open orders (something open we didn't create/track)
    - executedQty > 0 on any OPEN order (partial fill risk)
    - recent trades exist (unexpected fills), for far orders this should be ~0
    """
    open_orders = client.open_orders(symbol=symbol)
    open_ids = {int(o["orderId"]) for o in open_orders}

    missing = sorted(list(expected_open_order_ids - open_ids))
    extra = sorted(list(open_ids - expected_open_order_ids))

    any_exec = any(_safe_float(o.get("executedQty", "0")) > 0 for o in open_orders)

    trades_count: int | None = None
    if check_trades:
        try:
            trades = client.my_trades(symbol=symbol, limit=trades_limit)
            trades_count = len(trades) if isinstance(trades, list) else 0
        except Exception:
            trades_count = None

    ok = True
    reason = "ok"

    if any_exec:
        ok = False
        reason = "Unexpected executedQty>0 on OPEN order (partial fill risk)"

    if trades_count not in (None, 0):
        ok = False
        reason = f"Unexpected trades detected (count={trades_count})"

    if extra and ok:
        ok = False
        reason = f"Unexpected open orders present (count={len(extra)})"

    return OrderReconcileResult(
        ok=ok,
        open_orders_count=len(open_orders),
        open_order_ids=open_ids,
        expected_open_order_ids=set(expected_open_order_ids),
        missing_expected_open_orders=missing,
        unexpected_open_orders=extra,
        any_open_order_executed_qty_gt_0=any_exec,
        recent_trades_count=trades_count,
        reason=reason,
    )
    