from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass(frozen=True)
class BinancePublicConfig:
    base_url: str = "https://api.binance.com"
    timeout_seconds: int = 30
    min_request_interval_seconds: float = 0.15


class BinancePublicClient:
    def __init__(self, cfg: Optional[BinancePublicConfig] = None):
        self.cfg = cfg or BinancePublicConfig()
        self._last_request_ts = 0.0

    def _throttle(self) -> None:
        dt = time.time() - self._last_request_ts
        if dt < self.cfg.min_request_interval_seconds:
            time.sleep(self.cfg.min_request_interval_seconds - dt)

    def _get(self, path: str, params: Dict[str, Any]) -> Any:
        self._throttle()
        url = f"{self.cfg.base_url}{path}"
        resp = requests.get(url, params=params, timeout=self.cfg.timeout_seconds)
        self._last_request_ts = time.time()

        # Basic rate-limit/HTTP robustness
        if resp.status_code in (418, 429):
            # Too many requests / banned temporarily
            raise RuntimeError(f"Binance rate limited: {resp.status_code} {resp.text}")

        resp.raise_for_status()
        return resp.json()

    def klines(
        self,
        symbol: str,
        interval: str,
        start_time_ms: int,
        end_time_ms: Optional[int] = None,
        limit: int = 1000,
    ) -> List[List[Any]]:
        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time_ms,
            "limit": limit,
        }
        if end_time_ms is not None:
            params["endTime"] = end_time_ms
        return self._get("/api/v3/klines", params=params)
