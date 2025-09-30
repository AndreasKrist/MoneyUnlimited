# Impressive Trading Bot with ML

A sophisticated, production-quality trading bot using ensemble machine learning models for cryptocurrency and stock trading.

## 🎯 Project Goals

- Multi-model ensemble system (Random Forest, XGBoost, LSTM)
- Backtested Sharpe ratio > 1.0 or win rate > 55%
- Professional dashboard with real-time visualizations
- Outperforms buy-and-hold benchmark
- 100% free tools and data sources

## 🏗️ Project Structure

```
MoneyUnlimited/
├── config/              # Configuration files
│   └── config.py        # Central configuration
├── src/                 # Source code
│   ├── data/            # Data pipeline (Phase 1 ✓)
│   │   ├── fetcher.py   # Data fetching (yfinance, ccxt)
│   │   ├── indicators.py # Technical indicators
│   │   ├── database.py  # SQLite caching
│   │   └── pipeline.py  # Main pipeline orchestrator
│   ├── models/          # ML models (Phase 2)
│   ├── backtesting/     # Backtesting engine (Phase 3)
│   └── dashboard/       # Streamlit dashboard (Phase 4)
├── data/                # SQLite database
├── logs/                # Log files
├── tests/               # Unit tests
└── requirements.txt     # Python dependencies
```

## 🚀 Phase 1: Data Infrastructure (COMPLETED)

### Features

✅ **Multi-source data fetching**
- Crypto data via `ccxt` (Binance)
- Stock data via `yfinance`
- Configurable symbols and timeframes

✅ **Technical indicators**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- EMA (Exponential Moving Average) - 9, 21, 50, 200 periods
- ATR (Average True Range)
- Volume indicators (OBV, Volume SMA, Volume Ratio)
- Price momentum and volatility features

✅ **SQLite database caching**
- Efficient storage of OHLCV data
- Separate indicator storage
- Incremental updates (no unnecessary re-fetching)

✅ **Modular, clean architecture**
- Type hints and docstrings
- Error handling
- Configurable via `config.py`

## 📦 Installation

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure settings (optional)

Edit `config/config.py` to customize:
- Data sources (crypto/stocks)
- Symbols and timeframes
- Indicator parameters
- Lookback period

## 🎮 Usage

### Run the complete data pipeline

```bash
python src/data/pipeline.py
```

This will:
1. Fetch historical data for all configured symbols
2. Calculate technical indicators
3. Store everything in SQLite database

### Command-line options

```bash
# Force refresh all data from scratch
python src/data/pipeline.py --refresh

# Show data summary only
python src/data/pipeline.py --summary

# Update specific symbol
python src/data/pipeline.py --symbol BTC/USDT --timeframe 1h --source crypto
```

### Use in Python code

```python
from src.data import DataPipeline

# Initialize pipeline
pipeline = DataPipeline()

# Update all configured data
pipeline.update_all_configured()

# Get data with indicators
df = pipeline.get_data_with_indicators('BTC/USDT', '1h')
print(df.tail())

# Close database connection
pipeline.close()
```

## 🔧 Configuration

Default configuration in `config/config.py`:

- **Crypto**: BTC/USDT on 1h and 4h timeframes
- **Stocks**: SPY on 1h and 1d timeframes
- **History**: 730 days (2 years)
- **Indicators**: All enabled with standard parameters

## 📊 Data Schema

### OHLCV Data
- timestamp, open, high, low, close, volume
- symbol, timeframe, source

### Technical Indicators
- RSI, MACD (line, signal, histogram)
- Bollinger Bands (upper, middle, lower, bandwidth)
- EMA (9, 21, 50, 200 periods)
- ATR, OBV
- Returns, momentum, volatility features

## 🛣️ Next Phases

- **Phase 2**: ML model training (Random Forest, XGBoost, LSTM)
- **Phase 3**: Backtesting engine with realistic simulation
- **Phase 4**: Streamlit dashboard with live visualizations
- **Phase 5**: Optional sentiment analysis integration

## 📝 Notes

- This bot is for demonstration/portfolio purposes only
- Backtested performance ≠ future results
- All tools and data sources are 100% free
- No API keys required for basic usage

## 🔐 Requirements

- Python 3.8+
- Internet connection for data fetching
- ~500MB disk space for 2 years of data

## 📄 License

MIT License - Feel free to use for learning and portfolio projects
