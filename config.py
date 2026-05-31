import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    bybit_base_url: str = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")
    bybit_category: str = os.getenv("BYBIT_CATEGORY", "linear")
    bybit_symbol: str = os.getenv("BYBIT_SYMBOL", "BTCUSDT")
    bybit_interval: str = os.getenv("BYBIT_INTERVAL", "60")
    default_rows: int = int(os.getenv("DEFAULT_ROWS", "2500"))
    default_horizon: int = int(os.getenv("DEFAULT_HORIZON", "6"))
    max_rows: int = int(os.getenv("MAX_ROWS", "5000"))
    secret_key: str = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")


config = AppConfig()
