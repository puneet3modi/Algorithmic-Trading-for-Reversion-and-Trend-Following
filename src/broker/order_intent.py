from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


Side = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: Side
    quantity: str          # will be quantized in live loop
    price: str             # will be quantized in live loop
    tif: str
    client_order_id: str
    reason: str
    target_position: int   # for logging only: 0/1 in spot usage


def far_limit_price(last_px: float, side: Side, far_bps: float) -> float:
    """
    far_bps=500 => 5% away from market.
    BUY far below, SELL far above.
    """
    bps = far_bps / 10000.0
    if side == "BUY":
        return last_px * (1.0 - bps)
    return last_px * (1.0 + bps)


def notional_to_qty(notional_usdt: float, last_px: float) -> float:
    return max(0.0, notional_usdt / max(last_px, 1e-12))


def decide_order(
    symbol: str,
    last_px: float,
    desired_position: int,
    current_position: int,
    notional_usdt: float,
    far_bps: float,
    now_ms: int,
    *,
    spot_mode: bool = True,
) -> Optional[OrderIntent]:
    """
    Spot-safe order intent:
    - desired_position is expected in {0,1} for spot.
    - If spot_mode=True, treat any negative desired as 0 (flat).

    We only place an order if desired != current.
    We use far-from-market LIMIT orders so fills are unlikely (safety).
    """
    if spot_mode and desired_position < 0:
        desired_position = 0

    # Nothing to do if we already match
    if desired_position == current_position:
        return None

    # Spot: flat means "no BUY". SELL is handled by the live loop only if holding base.
    if desired_position == 0:
        side: Side = "SELL"
    else:
        side = "BUY"

    px = far_limit_price(last_px, side, far_bps)
    qty = notional_to_qty(notional_usdt, last_px)

    qty_s = f"{qty:.8f}"
    px_s = f"{px:.8f}"

    cid = f"cqf_m6_{symbol}_{now_ms}_{side}"
    reason = f"spot_mode={spot_mode} desired={desired_position} current={current_position} far_bps={far_bps}"

    return OrderIntent(
        symbol=symbol,
        side=side,
        quantity=qty_s,
        price=px_s,
        tif="GTC",
        client_order_id=cid,
        reason=reason,
        target_position=desired_position,
    )
    