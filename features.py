from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    high = df["high"]
    low = df["low"]

    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = (tenkan + kijun) / 2
    span_b = (high.rolling(52).max() + low.rolling(52).min()) / 2

    df["ichimoku_tenkan"] = tenkan
    df["ichimoku_kijun"] = kijun
    df["ichimoku_span_a"] = span_a
    df["ichimoku_span_b"] = span_b
    df["cloud_thickness"] = (span_a - span_b).abs()
    df["close_to_tenkan"] = df["close"] / tenkan - 1
    df["close_to_kijun"] = df["close"] / kijun - 1
    df["above_cloud"] = (df["close"] > np.maximum(span_a, span_b)).astype(int)
    df["below_cloud"] = (df["close"] < np.minimum(span_a, span_b)).astype(int)
    return df


def make_features(raw: pd.DataFrame, horizon: int = 6) -> tuple[pd.DataFrame, list[str]]:
    df = raw.copy().sort_values("timestamp").reset_index(drop=True)

    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_6"] = df["close"].pct_change(6)
    df["return_12"] = df["close"].pct_change(12)
    df["log_return_1"] = np.log(df["close"]).diff(1)

    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["sma10_distance"] = df["close"] / df["sma_10"] - 1
    df["sma20_distance"] = df["close"] / df["sma_20"] - 1
    df["sma50_distance"] = df["close"] / df["sma_50"] - 1

    df["volatility_12"] = df["return_1"].rolling(12).std()
    df["volatility_24"] = df["return_1"].rolling(24).std()
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["body_pct"] = (df["close"] - df["open"]) / df["open"]
    df["volume_change"] = df["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
    df["rsi_14"] = rsi(df["close"], 14)

    df = add_ichimoku(df)

    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["future_close"] = df["close"].shift(-horizon)
    df["target"] = (df["future_close"] > df["close"]).astype(int)

    feature_cols = [
        "return_1", "return_3", "return_6", "return_12", "log_return_1",
        "sma10_distance", "sma20_distance", "sma50_distance",
        "macd", "macd_signal", "volatility_12", "volatility_24",
        "range_pct", "body_pct", "volume_change", "volume_ratio", "rsi_14",
        "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_span_a", "ichimoku_span_b",
        "cloud_thickness", "close_to_tenkan", "close_to_kijun", "above_cloud", "below_cloud",
        "hour_sin", "hour_cos", "dayofweek",
    ]

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols + ["target"])
    return df.reset_index(drop=True), feature_cols
