from __future__ import annotations

import hashlib
from urllib.parse import urlencode
import hmac
import math
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from typing import Any

import requests
import random

_RETRY_STATUS = {418, 429, 500, 502, 503, 504}

@dataclass(frozen=True)
class BinanceCredentials:
    api_key: str
    api_secret: str


@dataclass(frozen=True)
class SymbolFilters:
    tick_size: float
    step_size: float
    min_qty: float
    max_qty: float
    min_notional: float | None
    tick_size_s: str
    step_size_s: str


def _d(x: Any) -> Decimal:
    return Decimal(str(x))


def _quantize_down(x: Decimal, step: Decimal) -> Decimal:
    # floor to step: floor(x/step)*step
    if step <= 0:
        return x
    n = (x / step).to_integral_value(rounding=ROUND_DOWN)
    return n * step


def _quantize_nearest(x: Decimal, tick: Decimal) -> Decimal:
    # round to nearest tick
    if tick <= 0:
        return x
    n = (x / tick).to_integral_value(rounding=ROUND_HALF_UP)
    return n * tick


def _to_fixed_str(x: Decimal) -> str:
    # fixed-point, never scientific notation
    s = format(x, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


class BinanceREST:
    """
    Minimal Binance Spot REST client for Testnet.
    - Public endpoints: time, ticker_price, klines, exchangeInfo
    - Signed endpoints: account, open_orders, new_limit_order, cancel_order

    Key implementation detail:
    - Quantity/price MUST be sent as properly formatted fixed-point strings
    respecting LOT_SIZE.stepSize and PRICE_FILTER.tickSize (Binance rejects excess precision).
    """

    def __init__(
        self,
        base_url: str,
        creds: BinanceCredentials,
        recv_window_ms: int = 5000,
        timeout_s: int = 10,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.creds = creds
        self.recv_window_ms = int(recv_window_ms)
        self.timeout_s = int(timeout_s)

        self._session = requests.Session()
        self._session.headers.update({"X-MBX-APIKEY": self.creds.api_key})

        self._filters_cache: dict[str, SymbolFilters] = {}

    def _sign(self, query_string: str) -> str:
        sig = hmac.new(
            self.creds.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return sig

    def _request(self, method: str, path: str, params: dict | None = None, signed: bool = False) -> dict:
        url = f"{self.base_url}{path}"
        params = params or {}

        max_retries = 4
        base_sleep = 0.5

        for attempt in range(max_retries + 1):
            try:
                if signed:
                    params["timestamp"] = int(time.time() * 1000)
                    params["recvWindow"] = self.recv_window_ms

                    qs = urlencode(params, doseq=True)
                    sig = self._sign(qs)
                    qs_signed = qs + f"&signature={sig}"
                    full_url = url + "?" + qs_signed

                    resp = self._session.request(method, full_url, timeout=self.timeout_s)
                else:
                    resp = self._session.request(method, url, params=params, timeout=self.timeout_s)

                # Parse JSON (or fall back to raw text)
                try:
                    data = resp.json()
                except Exception:
                    data = {"raw": resp.text}

                # Retry on transient errors
                if resp.status_code in _RETRY_STATUS:
                    if attempt < max_retries:
                        sleep = base_sleep * (2 ** attempt) + random.random() * 0.2
                        time.sleep(sleep)
                        continue

                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code} {method} {path} -> {data}")

                if isinstance(data, dict):
                    return data
                return {"data": data}

            except Exception as e:
                # Retry on network-ish errors
                if attempt < max_retries:
                    sleep = base_sleep * (2 ** attempt) + random.random() * 0.2
                    time.sleep(sleep)
                    continue
                raise

    # Public endpoints
    def time(self) -> dict:
        return self._request("GET", "/api/v3/time", signed=False)

    def ticker_price(self, symbol: str) -> dict:
        return self._request("GET", "/api/v3/ticker/price", params={"symbol": symbol}, signed=False)

    def klines(self, symbol: str, interval: str, limit: int = 2) -> list:
        out = self._request(
            "GET",
            "/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": int(limit)},
            signed=False,
        )
        return out["data"]

    def exchange_info(self, symbol: str | None = None) -> dict:
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", "/api/v3/exchangeInfo", params=params, signed=False)

    # Filters + quantization
    def symbol_filters(self, symbol: str) -> SymbolFilters:
        if symbol in self._filters_cache:
            return self._filters_cache[symbol]

        info = self.exchange_info()
        sym = None
        for s in info.get("symbols", []):
            if s.get("symbol") == symbol:
                sym = s
                break
        if sym is None:
            raise RuntimeError(f"Symbol not found in exchangeInfo: {symbol}")

        tick_size = step_size = min_qty = max_qty = None
        tick_size_s = step_size_s = None
        min_notional = None

        for f in sym.get("filters", []):
            ft = f.get("filterType")
            if ft == "PRICE_FILTER":
                tick_size_s = f["tickSize"]
                tick_size = float(tick_size_s)
            elif ft == "LOT_SIZE":
                step_size_s = f["stepSize"]
                step_size = float(step_size_s)
                min_qty = float(f["minQty"])
                max_qty = float(f["maxQty"])
            elif ft in ("MIN_NOTIONAL", "NOTIONAL"):
                mn = f.get("minNotional")
                if mn is not None:
                    min_notional = float(mn)

        if tick_size is None or step_size is None or min_qty is None or max_qty is None:
            raise RuntimeError(
                f"Missing filters for {symbol}: tick={tick_size} step={step_size} minQty={min_qty} maxQty={max_qty}"
            )
        if tick_size_s is None or step_size_s is None:
            raise RuntimeError(f"Missing raw tick/step strings for {symbol}")

        filt = SymbolFilters(
            tick_size=tick_size,
            step_size=step_size,
            min_qty=min_qty,
            max_qty=max_qty,
            min_notional=min_notional,
            tick_size_s=tick_size_s,
            step_size_s=step_size_s,
        )
        self._filters_cache[symbol] = filt
        return filt

    def quantize_order(self, symbol: str, quantity: float, price: float) -> tuple[str, str]:
        """
        Returns (quantity_str, price_str) formatted to exactly valid precision for Binance.
        Uses Decimal to avoid float artifacts.
        """
        f = self.symbol_filters(symbol)

        step = _d(f.step_size_s)
        tick = _d(f.tick_size_s)

        q = _quantize_down(_d(quantity), step)
        p = _quantize_nearest(_d(price), tick)

        if q < _d(f.min_qty):
            raise RuntimeError(f"Quantity {q} < minQty {f.min_qty} for {symbol} (step={f.step_size_s})")
        if q > _d(f.max_qty):
            raise RuntimeError(f"Quantity {q} > maxQty {f.max_qty} for {symbol}")

        if f.min_notional is not None:
            notional = q * p
            if notional < _d(f.min_notional):
                raise RuntimeError(f"Notional {notional} < minNotional {f.min_notional} for {symbol}")

        return _to_fixed_str(q), _to_fixed_str(p)

    # Signed endpoints
    def account(self) -> dict:
        return self._request("GET", "/api/v3/account", signed=True)

    def open_orders(self, symbol: str | None = None) -> list[dict]:
        params = {}
        if symbol is not None:
            params["symbol"] = symbol
        out = self._request("GET", "/api/v3/openOrders", params=params, signed=True)
        return out["data"]

    def new_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: str,
        price: str,
        time_in_force: str = "GTC",
        new_client_order_id: str | None = None,
    ) -> dict:
        params = {
            "symbol": symbol,
            "side": side,
            "type": "LIMIT",
            "timeInForce": time_in_force,
            # critical: keep as strings
            "quantity": str(quantity),
            "price": str(price),
        }
        if new_client_order_id:
            params["newClientOrderId"] = new_client_order_id

        return self._request("POST", "/api/v3/order", params=params, signed=True)

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        return self._request(
            "DELETE",
            "/api/v3/order",
            params={"symbol": symbol, "orderId": int(order_id)},
            signed=True,
        )
        
    def my_trades(self, symbol: str, limit: int = 10) -> list:
        return self._request(
            "GET",
            "/api/v3/myTrades",
            params={"symbol": symbol, "limit": int(limit)},
            signed=True,
        )
        