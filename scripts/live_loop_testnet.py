from __future__ import annotations

from email import parser
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import time
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

from src.common.logging import setup_logger
from src.broker.binance_testnet import BinanceCredentials, BinanceREST
from src.broker.order_intent import decide_order
from src.broker.reconcile_orders import reconcile_open_orders_and_trades
from src.broker.reconcile import (
    infer_shadow_position_spot,
    reconcile_desired_vs_shadow_spot,
    should_trade,
)
from src.broker.reconcile_log import append_json_event


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _quantize_down(x: float, step: float) -> float:
    return float(np.floor(x / step) * step)


def _decimals_from_step(step: float) -> int:
    s = f"{step:.10f}".rstrip("0")
    if "." not in s:
        return 0
    return len(s.split(".")[1])


def _format_qty(qty: float, step: float) -> str:
    dec = _decimals_from_step(step)
    return f"{qty:.{dec}f}"


def _format_px(px: float, tick: float) -> str:
    dec = _decimals_from_step(tick)
    return f"{px:.{dec}f}"


def _append_event(
    path: str,
    event: str,
    symbol: str,
    last_px: float,
    prev_close: float,
    prev_bar_open_time_utc: str,
    desired_position: int,
    current_position: int,
    extra: dict | None = None,
) -> None:
    order_id = None
    order_status = None
    side = None
    price = None
    quantity = None

    if isinstance(extra, dict):
        order_id = extra.get("orderId")
        order_status = extra.get("status")
        side = extra.get("side")
        price = extra.get("price")
        quantity = extra.get("origQty") or extra.get("quantity")

    row = {
        "ts_utc": utc_now_iso(),
        "event": event,
        "symbol": symbol,
        "last_px": last_px,
        "prev_close": prev_close,
        "prev_bar_open_time_utc": prev_bar_open_time_utc,
        "desired_position": desired_position,
        "current_position": current_position,
        "order_id": order_id,
        "order_status": order_status,
        "side": side,
        "order_price": price,
        "order_qty": quantity,
        "extra": extra or {},
    }
    df = pd.DataFrame([row])
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


def main() -> None:
    logger = setup_logger("live_loop_testnet")
    load_dotenv()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run exactly one loop iteration and exit.")
    args = parser.parse_args()

    with open("config/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    base_url = cfg["broker"]["base_url_spot"]
    recv_window_ms = cfg["broker"].get("recv_window_ms", 5000)
    timeout_s = cfg["broker"].get("timeout_s", 10)

    symbol = cfg["live"]["symbol"]
    interval = cfg["live"]["interval"]
    sleep_s = int(cfg["live"]["loop_sleep_seconds"])
    notional = float(cfg["live"]["order_notional_usdt"])
    far_bps = float(cfg["live"]["far_bps"])
    max_open = int(cfg["live"]["max_open_orders"])
    cancel_after_min = int(cfg["live"]["cancel_stale_after_minutes"])

    reconcile_every = int(cfg["live"].get("reconcile_every_n_loops", 1))
    min_notional_usdt = float(cfg["live"].get("min_notional_usdt", 5.0))

    api_key = os.getenv("BINANCE_TESTNET_API_KEY", "")
    api_secret = os.getenv("BINANCE_TESTNET_API_SECRET", "")
    if not api_key or not api_secret:
        raise RuntimeError("Missing BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET in .env")

    client = BinanceREST(
        base_url=base_url,
        creds=BinanceCredentials(api_key=api_key, api_secret=api_secret),
        recv_window_ms=recv_window_ms,
        timeout_s=timeout_s,
    )

    # Exchange filters for LOT_SIZE / PRICE_FILTER / MIN_NOTIONAL
    info = client.exchange_info(symbol=symbol)
    sym = info["symbols"][0]
    filters = {f["filterType"]: f for f in sym["filters"]}

    tick_size = float(filters["PRICE_FILTER"]["tickSize"])
    step_size = float(filters["LOT_SIZE"]["stepSize"])
    min_qty = float(filters["LOT_SIZE"]["minQty"])
    min_notional = float(filters.get("MIN_NOTIONAL", {}).get("minNotional", 0.0) or 0.0)

    logger.info(
        f"Starting Testnet loop | symbol={symbol} interval={interval} base_url={base_url} "
        f"far_bps={far_bps} notional={notional} sleep_s={sleep_s}"
    )
    logger.info(
        f"Symbol filters: tickSize={tick_size} stepSize={step_size} minQty={min_qty} minNotional={min_notional}"
    )

    os.makedirs("data/live", exist_ok=True)
    out_path = "data/live/order_events_testnet.csv"

    expected_open_order_ids: set[int] = set()
    loop_i = 0

    # Smoke check connectivity
    server_time = client.time()
    logger.info(f"Binance time: {server_time}")

    while True:
        loop_i += 1
        t0 = time.time()
        now_ms = int(time.time() * 1000)

        # Defaults for error handler
        last_px = float("nan")
        prev_close = float("nan")
        prev_bar_open_time_utc = ""
        desired_position = 0
        shadow_position = 0

        try:
            # 1) Market data
            ticker = client.ticker_price(symbol)
            last_px = float(ticker["price"])

            # 2) Latest closed kline
            k = client.klines(symbol, interval, limit=2)
            prev_bar = k[-2]
            prev_close = float(prev_bar[4])
            prev_bar_open_time_utc = pd.to_datetime(int(prev_bar[0]), unit="ms", utc=True).isoformat()

            # 3) Strategy signal (EMA Ratio) from pipeline output
            ema_pos_path = f"data/processed/{symbol}_{interval}_ema_ratio_positions.csv"
            ema_df = pd.read_csv(ema_pos_path)
            ema_df["open_time"] = pd.to_datetime(ema_df["open_time"], utc=True)
            ema_df = ema_df.sort_values("open_time")
            desired_position = int(ema_df["position_ema_ratio"].iloc[-1])

            # 4) Open orders (broker truth)
            open_orders = client.open_orders(symbol=symbol)

            # 5) Account -> shadow position (balances)
            if reconcile_every <= 1 or (loop_i % reconcile_every == 0):
                acct = client.account()
                rec = infer_shadow_position_spot(
                    symbol=symbol,
                    account=acct,
                    last_px=last_px,
                    desired_position=desired_position,
                    notional_usdt=notional,
                    min_notional_usdt=min_notional_usdt,
                )
                shadow_position = rec.shadow_position
            else:
                shadow_position = 0  # conservative fallback

            # Spot target: in spot we map desired {-1,0,1} -> target {0,1}
            target_position = reconcile_desired_vs_shadow_spot(desired_position, shadow_position)

            logger.info(
                f"tick last_px={last_px:.2f} | prev_close={prev_close:.2f} | prev_bar_open={prev_bar_open_time_utc} | "
                f"desired={desired_position} shadow={shadow_position} target={target_position} open_orders={len(open_orders)}"
            )

            # Snapshot for report evidence
            append_json_event(
                "data/live/reconcile_snapshots_testnet.csv",
                {
                    "symbol": symbol,
                    "last_px": last_px,
                    "prev_close": prev_close,
                    "prev_bar_open_time_utc": prev_bar_open_time_utc,
                    "desired_position": desired_position,
                    "shadow_position": shadow_position,
                    "target_position": target_position,
                    "open_orders_count": len(open_orders),
                },
            )

            # 6) Cancel extras/stale
            if len(open_orders) > max_open:
                logger.warning(f"Too many open orders ({len(open_orders)}). Cancelling all to reset state.")
                for o in open_orders:
                    try:
                        client.cancel_order(symbol=symbol, order_id=int(o["orderId"]))
                        _append_event(
                            out_path,
                            "CANCEL",
                            symbol,
                            last_px,
                            prev_close,
                            prev_bar_open_time_utc,
                            desired_position,
                            shadow_position,
                            extra=o,
                        )
                    except Exception as e:
                        logger.warning(f"Cancel failed: {e}")

            cutoff_ms = now_ms - cancel_after_min * 60 * 1000
            for o in open_orders:
                o_time = int(o.get("time", now_ms))
                if o_time < cutoff_ms:
                    try:
                        client.cancel_order(symbol=symbol, order_id=int(o["orderId"]))
                        _append_event(
                            out_path,
                            "CANCEL_STALE",
                            symbol,
                            last_px,
                            prev_close,
                            prev_bar_open_time_utc,
                            desired_position,
                            shadow_position,
                            extra=o,
                        )
                    except Exception as e:
                        logger.warning(f"Cancel stale failed: {e}")

            # refresh after cancels
            open_orders = client.open_orders(symbol=symbol)

            # 7) Decide & (maybe) place a far order
            ok_trade, why = should_trade(target_position, shadow_position, len(open_orders))
            if ok_trade:
                # For spot:
                # - target 1 means BUY (go long)
                # - target 0 means SELL only if you actually hold base (shadow_position==1)
                effective_desired = 1 if target_position == 1 else -1

                intent = decide_order(
                    symbol=symbol,
                    last_px=last_px,
                    desired_position=effective_desired,
                    current_position=shadow_position,
                    notional_usdt=notional,
                    far_bps=far_bps,
                    now_ms=now_ms,
                )

                if intent is not None:
                    raw_qty = float(intent.quantity)
                    raw_px = float(intent.price)

                    qty_q = _quantize_down(raw_qty, step_size)
                    px_q = _quantize_down(raw_px, tick_size)

                    if qty_q < min_qty:
                        _append_event(
                            out_path,
                            "SKIP_ORDER_MINQTY",
                            symbol,
                            last_px,
                            prev_close,
                            prev_bar_open_time_utc,
                            desired_position,
                            shadow_position,
                            extra={"raw_qty": raw_qty, "qty_q": qty_q, "min_qty": min_qty},
                        )
                        logger.warning(f"qty below minQty: {qty_q} < {min_qty}; skipping order.")
                    elif min_notional > 0 and (qty_q * last_px) < min_notional:
                        _append_event(
                            out_path,
                            "SKIP_ORDER_MINNOTIONAL",
                            symbol,
                            last_px,
                            prev_close,
                            prev_bar_open_time_utc,
                            desired_position,
                            shadow_position,
                            extra={"qty_q": qty_q, "min_notional": min_notional, "notional": qty_q * last_px},
                        )
                        logger.warning(
                            f"notional below minNotional: {qty_q*last_px:.2f} < {min_notional}; skipping order."
                        )
                    else:
                        qty_str = _format_qty(qty_q, step_size)
                        px_str = _format_px(px_q, tick_size)

                        logger.info(
                            f"Quantized: raw_qty={raw_qty:.8f} -> {qty_str} | raw_px={raw_px:.2f} -> {px_str}"
                        )

                        resp = client.new_limit_order(
                            symbol=intent.symbol,
                            side=intent.side,
                            quantity=qty_str,
                            price=px_str,
                            time_in_force=intent.tif,
                            new_client_order_id=intent.client_order_id,
                        )

                        try:
                            expected_open_order_ids.add(int(resp.get("orderId")))
                        except Exception:
                            pass

                        _append_event(
                            out_path,
                            "NEW_ORDER",
                            symbol,
                            last_px,
                            prev_close,
                            prev_bar_open_time_utc,
                            desired_position,
                            shadow_position,
                            extra=resp,
                        )
                        logger.info(
                            f"Submitted {intent.side} LIMIT qty={qty_str} @ {px_str} cid={intent.client_order_id}"
                        )
            else:
                _append_event(
                    out_path,
                    "SKIP",
                    symbol,
                    last_px,
                    prev_close,
                    prev_bar_open_time_utc,
                    desired_position,
                    shadow_position,
                    extra={"reason": why},
                )

            # 8) Snapshot open orders after action
            open_orders = client.open_orders(symbol=symbol)
            _append_event(
                out_path,
                "OPEN_ORDERS_SNAPSHOT",
                symbol,
                last_px,
                prev_close,
                prev_bar_open_time_utc,
                desired_position,
                shadow_position,
                extra={"open_orders_count": len(open_orders)},
            )

            # 9) Broker reconciliation (orders + trades)
            rec_orders = reconcile_open_orders_and_trades(
                client,
                symbol=symbol,
                expected_open_order_ids=expected_open_order_ids,
                check_trades=True,
                trades_limit=5,
            )

            _append_event(
                out_path,
                "RECONCILE_OK" if rec_orders.ok else "RECONCILE_FAIL",
                symbol,
                last_px,
                prev_close,
                prev_bar_open_time_utc,
                desired_position,
                shadow_position,
                extra={
                    "ok": rec_orders.ok,
                    "reason": rec_orders.reason,
                    "open_orders_count": rec_orders.open_orders_count,
                    "missing_expected_open_orders": rec_orders.missing_expected_open_orders,
                    "unexpected_open_orders": rec_orders.unexpected_open_orders,
                    "any_open_order_executed_qty_gt_0": rec_orders.any_open_order_executed_qty_gt_0,
                    "recent_trades_count": rec_orders.recent_trades_count,
                },
            )

            if not rec_orders.ok:
                logger.error(f"RECONCILE_FAIL: {rec_orders.reason}")

                # Safety: cancel unexpected orders if found
                if rec_orders.unexpected_open_orders:
                    logger.warning("Cancelling unexpected open orders for safety.")
                    for oid in rec_orders.unexpected_open_orders:
                        try:
                            client.cancel_order(symbol=symbol, order_id=int(oid))
                        except Exception as e:
                            logger.warning(f"Cancel unexpected orderId={oid} failed: {e}")

        except Exception as e:
            logger.exception(f"Loop error: {e}")
            _append_event(
                out_path,
                "ERROR",
                symbol,
                last_px,
                prev_close,
                prev_bar_open_time_utc,
                desired_position,
                shadow_position,
                extra={"error": str(e)},
            )

        # If running once, exit immediately (no sleeping)
        if args.once:
            logger.info("Ran once (--once). Exiting.")
            return

        elapsed = time.time() - t0
        time.sleep(max(0.0, sleep_s - elapsed))


if __name__ == "__main__":
    main()
    