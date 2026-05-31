from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from features import make_features
from pdt import PermutationDecisionTreeClassifier


@dataclass
class ForecastResult:
    prediction_label: str
    prediction_value: int
    confidence: float
    current_price: float
    horizon: int
    source: str
    source_details: str
    fallback_reason: str | None
    metrics: dict[str, Any]
    baselines: dict[str, Any]
    chart_data: dict[str, Any]
    feature_importance: list[dict[str, Any]]
    rules: list[str]
    rows_used: int
    last_timestamp: str


def _safe_auc(y_true, p1) -> float | None:
    try:
        return float(roc_auc_score(y_true, p1))
    except Exception:
        return None


def _metrics(y_true, y_pred, p1=None) -> dict[str, float | None]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(_safe_auc(y_true, p1), 4) if p1 is not None and _safe_auc(y_true, p1) is not None else None,
    }


def _time_split(df: pd.DataFrame, feature_cols: list[str], test_size: float = 0.25):
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    X_train = train[feature_cols].to_numpy()
    y_train = train["target"].to_numpy(dtype=int)
    X_test = test[feature_cols].to_numpy()
    y_test = test["target"].to_numpy(dtype=int)
    return train, test, X_train, X_test, y_train, y_test


def train_and_forecast(raw_df: pd.DataFrame, horizon: int = 6) -> ForecastResult:
    df, feature_cols = make_features(raw_df, horizon=horizon)
    if len(df) < 400:
        raise ValueError("Not enough rows after feature engineering. Use a larger dataset.")

    train, test, X_train, X_test, y_train, y_test = _time_split(df, feature_cols)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    latest_scaled = scaler.transform(df[feature_cols].tail(1).to_numpy())

    pdt = PermutationDecisionTreeClassifier(max_depth=4, min_samples_leaf=35, n_thresholds=9)
    pdt.fit(X_train_scaled, y_train, feature_names=feature_cols)
    pdt_pred = pdt.predict(X_test_scaled)
    pdt_proba = pdt.predict_proba(X_test_scaled)[:, 1]
    latest_proba = float(pdt.predict_proba(latest_scaled)[0, 1])
    latest_prediction = int(latest_proba >= 0.5)

    decision_tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=35, random_state=42)
    decision_tree.fit(X_train_scaled, y_train)
    dt_pred = decision_tree.predict(X_test_scaled)
    dt_proba = decision_tree.predict_proba(X_test_scaled)[:, 1]

    boosting = GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=3, random_state=42)
    boosting.fit(X_train_scaled, y_train)
    gb_pred = boosting.predict(X_test_scaled)
    gb_proba = boosting.predict_proba(X_test_scaled)[:, 1]

    metrics = _metrics(y_test, pdt_pred, pdt_proba)
    baselines = {
        "Decision Tree": _metrics(y_test, dt_pred, dt_proba),
        "Gradient Boosting": _metrics(y_test, gb_pred, gb_proba),
    }

    importances = []
    for name, value in sorted(zip(feature_cols, boosting.feature_importances_), key=lambda x: x[1], reverse=True)[:10]:
        importances.append({"feature": name, "importance": round(float(value), 4)})

    chart_df = df.tail(220).copy()
    chart_data = {
        "timestamp": chart_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
        "open": chart_df["open"].round(2).tolist(),
        "high": chart_df["high"].round(2).tolist(),
        "low": chart_df["low"].round(2).tolist(),
        "close": chart_df["close"].round(2).tolist(),
        "sma20": chart_df["sma_20"].round(2).tolist(),
        "sma50": chart_df["sma_50"].round(2).tolist(),
    }

    source = raw_df.attrs.get("source", "Unknown source")
    source_details = raw_df.attrs.get("source_details", "")
    fallback_reason = raw_df.attrs.get("fallback_reason")
    label = "BTC is expected to grow" if latest_prediction == 1 else "BTC is expected to fall"
    confidence = latest_proba if latest_prediction == 1 else 1 - latest_proba

    return ForecastResult(
        prediction_label=label,
        prediction_value=latest_prediction,
        confidence=round(float(confidence), 4),
        current_price=round(float(df["close"].iloc[-1]), 2),
        horizon=horizon,
        source=source,
        source_details=source_details,
        fallback_reason=fallback_reason,
        metrics=metrics,
        baselines=baselines,
        chart_data=chart_data,
        feature_importance=importances,
        rules=pdt.rules(max_rules=10),
        rows_used=len(df),
        last_timestamp=str(df["timestamp"].iloc[-1]),
    )
