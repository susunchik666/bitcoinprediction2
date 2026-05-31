from __future__ import annotations

"""
Service layer for the Bitcoin forecasting web application.

This file intentionally keeps the application logic easy to read.  The route in
app.py calls run_forecast(), while the lower-level implementation is split into
small modules:
- data_loader.py downloads live BTCUSDT candles from Bybit API;
- features.py prepares technical indicators, including Ichimoku features;
- ml_pipeline.py trains PDT and baseline models and returns metrics;
- pdt.py contains the Permutation Decision Tree implementation.
"""

from dataclasses import asdict
from typing import Any

import pandas as pd

from data_loader import load_market_data, load_from_bybit_api
from ml_pipeline import ForecastResult, train_and_forecast


def fetch_candles(rows: int) -> pd.DataFrame:
    """Download recent BTCUSDT OHLCV candles from the live Bybit API."""
    return load_from_bybit_api(rows=rows)


def candles_to_dataframe(candles: pd.DataFrame) -> pd.DataFrame:
    """Return a clean DataFrame representation of downloaded candles.

    In the current implementation Bybit candles are already normalized in
    data_loader.py.  This function is kept as an explicit service step because it
    makes the pipeline clearer in the report and in the web application.
    """
    return candles.copy()


def train_and_predict(df: pd.DataFrame, horizon: int) -> ForecastResult:
    """Train the PDT model and baseline models, then return the latest forecast."""
    return train_and_forecast(df, horizon=horizon)


def run_forecast(horizon: int, rows: int, source: str = "bybit_api") -> dict[str, Any]:
    """Full end-to-end experiment used by the Flask route.

    Steps:
    1. download BTCUSDT candles through Bybit API;
    2. convert candles to a DataFrame;
    3. build technical features, including Ichimoku features;
    4. train and evaluate Permutation Decision Tree;
    5. return forecast, metrics, feature importance, rules and chart data.
    """
    raw_candles = load_market_data(source=source, rows=rows)
    df = candles_to_dataframe(raw_candles)
    result = train_and_predict(df, horizon=horizon)
    return asdict(result)
