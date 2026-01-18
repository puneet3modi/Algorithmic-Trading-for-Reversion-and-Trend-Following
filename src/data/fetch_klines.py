from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple, List

import pandas as pd

from src.data.binance_public import BinancePublicClient


def parse_iso_utc_to_ms(iso_utc: str) -> int:
    # expects e.g. "2022-01-01T00:00:00Z"
    dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
    return int(dt.timestamp() * 1000)


@dataclass(frozen=True)
class KlinesFetchSpec:
    symbol: str
    interval: str
    start_utc: str
    end_utc: str
    limit_per_request: int = 1000


KLINE_COLUMNS = [
    "open_time_ms",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_ms",
    "quote_asset_volume",
    "num_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def fetch_klines_to_df(client: BinancePublicClient, spec: KlinesFetchSpec) -> pd.DataFrame:
    start_ms = parse_iso_utc_to_ms(spec.start_utc)
    end_ms = parse_iso_utc_to_ms(spec.end_utc)

    all_rows: List[list] = []
    cursor = start_ms
    n_req = 0

    # Binance returns klines inclusive; we advance cursor to last_open_time + 1
    while cursor < end_ms:
        batch = client.klines(
            symbol=spec.symbol,
            interval=spec.interval,
            start_time_ms=cursor,
            end_time_ms=end_ms,
            limit=spec.limit_per_request,
        )
        n_req += 1
        if not batch:
            break

        all_rows.extend(batch)

        last_open_time = int(batch[-1][0])
        # move forward by 1ms to avoid duplicates
        cursor = last_open_time + 1

        # Safety stop: if API returns same last open time repeatedly
        if len(batch) == 1 and last_open_time == int(batch[0][0]):
            cursor += 1

    df = pd.DataFrame(all_rows, columns=KLINE_COLUMNS)

    # Convert types
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    int_cols = ["open_time_ms", "close_time_ms", "num_trades"]
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # Add timestamps
    df["open_time"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time_ms"], unit="ms", utc=True)

    # De-dup and sort
    df = df.drop_duplicates(subset=["open_time_ms"]).sort_values("open_time_ms").reset_index(drop=True)

    return df


def make_output_paths(raw_dir: str, symbol: str, interval: str) -> str:
    stem = f"{symbol}_{interval}_klines"
    return f"{raw_dir}/{stem}.csv"


def save_df(df: pd.DataFrame, csv_path: str) -> None:
    df.to_csv(csv_path, index=False)
