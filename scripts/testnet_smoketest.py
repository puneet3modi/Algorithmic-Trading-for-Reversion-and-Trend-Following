from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import yaml
from dotenv import load_dotenv

from src.common.logging import setup_logger
from src.broker.binance_testnet import BinanceCredentials, BinanceREST


def main() -> None:
    logger = setup_logger("testnet_smoketest")
    load_dotenv()

    with open("config/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    base_url = cfg["broker"]["base_url_spot"]
    symbol = cfg["live"]["symbol"]

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

    logger.info(f"time: {client.time()}")
    logger.info(f"ticker: {client.ticker_price(symbol)}")

    # Signed endpoints
    acct = client.account()
    logger.info(f"account canTrade={acct.get('canTrade')} balances_count={len(acct.get('balances', []))}")

    oo = client.open_orders(symbol=symbol)
    logger.info(f"open_orders_count={len(oo)}")


if __name__ == "__main__":
    main()
    