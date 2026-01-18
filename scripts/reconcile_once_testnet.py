from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import yaml
from dotenv import load_dotenv

from src.common.logging import setup_logger
from src.broker.binance_testnet import BinanceCredentials, BinanceREST
from src.broker.reconcile import infer_shadow_position_spot, reconcile_desired_vs_shadow_spot
from src.broker.reconcile_orders import reconcile_open_orders_and_trades


def main() -> None:
    logger = setup_logger("reconcile_once_testnet")
    load_dotenv()

    with open("config/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    symbol = cfg["live"]["symbol"]
    base_url = cfg["broker"]["base_url_spot"]

    api_key = os.getenv("BINANCE_TESTNET_API_KEY", "")
    api_secret = os.getenv("BINANCE_TESTNET_API_SECRET", "")
    if not api_key or not api_secret:
        raise RuntimeError("Missing BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET in .env")

    client = BinanceREST(
        base_url=base_url,
        creds=BinanceCredentials(api_key=api_key, api_secret=api_secret),
        recv_window_ms=cfg["broker"].get("recv_window_ms", 5000),
        timeout_s=cfg["broker"].get("timeout_s", 10),
    )

    # market
    last_px = float(client.ticker_price(symbol)["price"])

    # strategy desired (from latest EMA ratio file)
    interval = cfg["live"]["interval"]
    ema_path = f"data/processed/{symbol}_{interval}_ema_ratio_positions.csv"
    import pandas as pd

    ema = pd.read_csv(ema_path)
    ema["open_time"] = pd.to_datetime(ema["open_time"], utc=True)
    ema = ema.sort_values("open_time")
    desired = int(ema["position_ema_ratio"].iloc[-1])

    acct = client.account()
    rec = infer_shadow_position_spot(
        symbol=symbol,
        account=acct,
        last_px=last_px,
        desired_position=desired,
        notional_usdt=float(cfg["live"]["order_notional_usdt"]),
        min_notional_usdt=float(cfg["live"].get("min_notional_usdt", 5.0)),
    )
    target = reconcile_desired_vs_shadow_spot(desired, rec.shadow_position)

    # orders+trades reconcile
    open_orders = client.open_orders(symbol=symbol)
    expected = {int(o["orderId"]) for o in open_orders}  # baseline expectation for this one-shot
    ro = reconcile_open_orders_and_trades(
        client,
        symbol=symbol,
        expected_open_order_ids=expected,
        check_trades=True,
        trades_limit=10,
    )

    logger.info(
        f"last_px={last_px:.2f} desired={desired} shadow={rec.shadow_position} target={target} "
        f"open_orders={len(open_orders)} reconcile_ok={ro.ok} trades={ro.recent_trades_count}"
    )
    if not ro.ok:
        logger.error(f"reconcile_fail_reason={ro.reason}")


if __name__ == "__main__":
    main()
    