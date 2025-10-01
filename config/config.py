"""
Configuration file for trading bot
Centralized settings for data sources, trading parameters, and model configurations
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Database settings
DB_PATH = DATA_DIR / "trading_data.db"

# Data source settings
DATA_SOURCES = {
    "crypto": {
        "enabled": True,  # Temporarily disabled - install ccxt to enable
        "exchange": "binance",  # Using ccxt
        "symbols": ["BTC/USDT"],
        "timeframes": ["1h", "4h"],
    },
    "stocks": {
        "enabled": True,
        "symbols": ["SPY"],
        "timeframes": ["1h", "1d"],
    }
}

# Historical data settings
HISTORICAL_DATA = {
    "lookback_days": 730,  # 2 years of data
    "start_date": None,  # Optional: specific start date (YYYY-MM-DD)
    "end_date": None,  # Optional: specific end date (YYYY-MM-DD)
}

# Technical indicators settings
INDICATORS = {
    "rsi": {
        "enabled": True,
        "period": 14,
    },
    "macd": {
        "enabled": True,
        "fast": 12,
        "slow": 26,
        "signal": 9,
    },
    "bollinger_bands": {
        "enabled": True,
        "period": 20,
        "std": 2,
    },
    "ema": {
        "enabled": True,
        "periods": [9, 21, 50, 200],
    },
    "atr": {
        "enabled": True,
        "period": 14,
    },
    "volume": {
        "enabled": True,
        "sma_period": 20,
    }
}

# Trading strategy settings (for future phases)
TRADING = {
    "prediction_horizon": 4,  # Hours ahead to predict
    "position_size": 0.1,  # 10% of portfolio per trade
    "stop_loss": 0.02,  # 2%
    "take_profit": 0.04,  # 4%
    "max_positions": 3,
}

# Model settings (for future phases)
MODELS = {
    "ensemble": {
        "models": ["random_forest", "xgboost", "lstm"],
        "voting": "soft",  # soft or hard voting
    },
    "train_test_split": 0.8,
    "validation_split": 0.1,
}

# Logging settings
LOGGING = {
    "level": "INFO",
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    "rotation": "10 MB",
}

# API rate limiting (to respect free tier limits)
RATE_LIMITS = {
    "yfinance": {
        "requests_per_minute": 60,
    },
    "ccxt": {
        "requests_per_minute": 1200,  # Binance public API limit
    }
}
