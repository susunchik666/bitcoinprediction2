from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import pandas as pd
import requests

from config import config

DataSource = Literal["bybit_api"]

INTERVAL_TO_MINUTES = {
    "1": 1,
    "3": 3,
    "5": 5,
    "15": 15,
    "30": 30,
    "60": 60,
    "120": 120,
    "240": 240,
    "360": 360,
    "720": 720,
    "D": 1440,
    "W": 10080,
    "M": 43200,
}


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]

    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in market data: {missing}")

    keep = required + [column for column in ["turnover"] if column in df.columns]
    df = df[keep]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=required)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    return df.reset_index(drop=True)


def _request_bybit_kline(end_ms: int | None, limit: int) -> list[list[str]]:
    url = f"{config.bybit_base_url.rstrip('/')}/v5/market/kline"
    params = {
        "category": config.bybit_category,
        "symbol": config.bybit_symbol,
        "interval": config.bybit_interval,
        "limit": min(max(limit, 1), 1000),
    }
    if end_ms is not None:
        params["end"] = end_ms

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()

    if payload.get("retCode") != 0:
        message = payload.get("retMsg", "Unknown Bybit API error")
        raise RuntimeError(f"Bybit API error: {message}")

    result = payload.get("result") or {}
    rows = result.get("list") or []
    if not rows:
        raise RuntimeError("Bybit API returned no candle data")
    return rows


def load_from_bybit_api(rows: int | None = None) -> pd.DataFrame:
    rows = rows or config.default_rows
    rows = max(200, min(rows, config.max_rows))

    interval_minutes = INTERVAL_TO_MINUTES.get(str(config.bybit_interval), 60)
    step_ms = interval_minutes * 60 * 1000

    all_rows: list[list[str]] = []
    end_ms: int | None = None

    while len(all_rows) < rows:
        batch_size = min(1000, rows - len(all_rows))
        batch = _request_bybit_kline(end_ms=end_ms, limit=batch_size)
        all_rows.extend(batch)

        oldest_start_ms = min(int(item[0]) for item in batch)
        next_end_ms = oldest_start_ms - step_ms
        if end_ms == next_end_ms:
            break
        end_ms = next_end_ms

        if len(batch) < batch_size:
            break

    df = pd.DataFrame(
        all_rows,
        columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)

    normalized = _normalize_ohlcv(df).tail(rows).reset_index(drop=True)
    normalized.attrs["source"] = "Live Bybit API"
    normalized.attrs["source_details"] = f"{config.bybit_category}:{config.bybit_symbol}, interval={config.bybit_interval}"
    normalized.attrs["loaded_at"] = datetime.now(timezone.utc).isoformat()
    return normalized


def load_market_data(source: DataSource = "bybit_api", rows: int | None = None) -> pd.DataFrame:
    return load_from_bybit_api(rows=rows)
