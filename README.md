# BTC PDT Forecast

A Flask web application for short-term Bitcoin direction forecasting. The app downloads live BTCUSDT candlestick data from the Bybit public API, computes technical indicators, trains a Permutation Decision Tree model, compares it with baseline models, and shows the forecast in a visual dashboard.

The main prediction task is binary classification:

- **UP** — BTC price is expected to grow after the selected horizon;
- **DOWN** — BTC price is expected to fall or stay below the current price after the selected horizon.

The forecast is research-based and is not financial advice.

## Features

- Live BTCUSDT OHLCV data loading from Bybit API.
- Feature engineering with returns, volatility, moving averages, MACD, RSI and Ichimoku-based indicators.
- Custom Permutation Decision Tree classifier based on Effort-To-Compress split gain.
- Baseline comparison with Decision Tree and Gradient Boosting classifiers.
- Model evaluation using accuracy, F1-score, precision, recall and ROC-AUC.
- Interactive web dashboard with current BTC price, model confidence, candlestick chart, metrics, feature importance and PDT rules.

## Project structure

```text
bitcoinprediction2_api_live/
├── app.py                  # Flask routes
├── config.py               # Project settings
├── data_loader.py          # Live Bybit API data loading
├── services.py             # End-to-end forecasting service layer
├── features.py             # Technical indicators and ML features
├── pdt.py                  # Permutation Decision Tree implementation
├── ml_pipeline.py          # Training, testing, baselines and forecast result
├── requirements.txt        # Python dependencies
├── templates/              # HTML templates
└── static/                 # CSS styles
```

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python3 app.py
```

Open the application in a browser:

```text
http://127.0.0.1:5000
```

