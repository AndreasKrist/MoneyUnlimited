"""
Data fetcher module for retrieving OHLCV data from multiple sources
Supports yfinance (stocks) and ccxt (crypto)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Literal
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import DATA_SOURCES, HISTORICAL_DATA, RATE_LIMITS

try:
    import yfinance as yf
except ImportError:
    yf = None
    print("Warning: yfinance not installed. Stock data fetching disabled.")

try:
    import ccxt
except ImportError:
    ccxt = None
    print("Warning: ccxt not installed. Crypto data fetching disabled.")


class DataFetcher:
    """Fetches OHLCV data from various sources"""

    def __init__(self):
        """Initialize data fetcher with configured sources"""
        self.crypto_enabled = DATA_SOURCES["crypto"]["enabled"] and ccxt is not None
        self.stocks_enabled = DATA_SOURCES["stocks"]["enabled"] and yf is not None

        if self.crypto_enabled:
            exchange_id = DATA_SOURCES["crypto"]["exchange"]
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })

    def fetch_crypto_data(self, symbol: str, timeframe: str,
                         start_date: Optional[datetime] = None,
                         limit: int = 1000) -> pd.DataFrame:
        """
        Fetch crypto OHLCV data using ccxt

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1h', '4h', '1d')
            start_date: Start date for historical data
            limit: Maximum number of candles to fetch per request

        Returns:
            DataFrame with columns [timestamp, open, high, low, close, volume]
        """
        if not self.crypto_enabled:
            raise RuntimeError("Crypto data fetching is not enabled")

        # Convert start_date to milliseconds
        since = int(start_date.timestamp() * 1000) if start_date else None

        all_candles = []
        while True:
            try:
                # Fetch candles
                candles = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit
                )

                if not candles:
                    break

                all_candles.extend(candles)

                # Check if we got less than limit (means we reached the end)
                if len(candles) < limit:
                    break

                # Update since to last candle timestamp + 1ms
                since = candles[-1][0] + 1

                # Rate limiting
                time.sleep(self.exchange.rateLimit / 1000)

            except Exception as e:
                print(f"Error fetching crypto data: {e}")
                break

        if not all_candles:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        # Convert timestamp from milliseconds to seconds
        df['timestamp'] = df['timestamp'] // 1000

        return df

    def fetch_stock_data(self, symbol: str, timeframe: str,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch stock OHLCV data using yfinance

        Args:
            symbol: Stock ticker (e.g., 'SPY', 'AAPL')
            timeframe: Candle timeframe (e.g., '1h', '1d')
            start_date: Start date for historical data
            end_date: End date for historical data

        Returns:
            DataFrame with columns [timestamp, open, high, low, close, volume]
        """
        if not self.stocks_enabled:
            raise RuntimeError("Stock data fetching is not enabled")

        # Map timeframes to yfinance intervals
        interval_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '1d': '1d',
            '1wk': '1wk',
            '1mo': '1mo'
        }

        interval = interval_map.get(timeframe, '1h')

        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=HISTORICAL_DATA["lookback_days"])

        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )

            if df.empty:
                return pd.DataFrame()

            # Standardize column names
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Reset index to get timestamp as column
            df = df.reset_index()
            df = df.rename(columns={'Date': 'timestamp', 'Datetime': 'timestamp'})

            # Select only needed columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            # Convert timestamp to Unix seconds
            df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**9

            return df

        except Exception as e:
            print(f"Error fetching stock data for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_data(self, symbol: str, timeframe: str,
                   source: Literal['crypto', 'stocks'],
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Unified interface to fetch data from any source

        Args:
            symbol: Trading pair or stock ticker
            timeframe: Candle timeframe
            source: Data source ('crypto' or 'stocks')
            start_date: Start date for historical data
            end_date: End date for historical data

        Returns:
            DataFrame with OHLCV data
        """
        # Set default start date
        if not start_date:
            if HISTORICAL_DATA["start_date"]:
                start_date = datetime.strptime(HISTORICAL_DATA["start_date"], "%Y-%m-%d")
            else:
                lookback = HISTORICAL_DATA["lookback_days"]
                start_date = datetime.now() - timedelta(days=lookback)

        # Set default end date
        if not end_date and HISTORICAL_DATA["end_date"]:
            end_date = datetime.strptime(HISTORICAL_DATA["end_date"], "%Y-%m-%d")

        # Fetch based on source
        if source == 'crypto':
            df = self.fetch_crypto_data(symbol, timeframe, start_date)
        elif source == 'stocks':
            df = self.fetch_stock_data(symbol, timeframe, start_date, end_date)
        else:
            raise ValueError(f"Unknown source: {source}")

        # Basic data cleaning
        if not df.empty:
            df = df.dropna()
            df = df.drop_duplicates(subset=['timestamp'])
            df = df.sort_values('timestamp')

        return df

    def get_latest_price(self, symbol: str, source: Literal['crypto', 'stocks']) -> Optional[float]:
        """
        Get the latest price for a symbol

        Args:
            symbol: Trading pair or stock ticker
            source: Data source

        Returns:
            Latest close price or None
        """
        try:
            if source == 'crypto' and self.crypto_enabled:
                ticker = self.exchange.fetch_ticker(symbol)
                return ticker['last']
            elif source == 'stocks' and self.stocks_enabled:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='1d', interval='1m')
                if not data.empty:
                    return data['Close'].iloc[-1]
        except Exception as e:
            print(f"Error fetching latest price: {e}")

        return None


def fetch_all_configured_data(start_date: Optional[datetime] = None) -> dict:
    """
    Fetch data for all configured symbols and timeframes

    Args:
        start_date: Optional start date override

    Returns:
        Dictionary with data organized by source, symbol, and timeframe
    """
    fetcher = DataFetcher()
    results = {}

    # Fetch crypto data
    if DATA_SOURCES["crypto"]["enabled"]:
        results["crypto"] = {}
        for symbol in DATA_SOURCES["crypto"]["symbols"]:
            results["crypto"][symbol] = {}
            for timeframe in DATA_SOURCES["crypto"]["timeframes"]:
                print(f"Fetching crypto: {symbol} {timeframe}")
                df = fetcher.fetch_data(symbol, timeframe, 'crypto', start_date)
                results["crypto"][symbol][timeframe] = df
                print(f"  -> Fetched {len(df)} candles")

    # Fetch stock data
    if DATA_SOURCES["stocks"]["enabled"]:
        results["stocks"] = {}
        for symbol in DATA_SOURCES["stocks"]["symbols"]:
            results["stocks"][symbol] = {}
            for timeframe in DATA_SOURCES["stocks"]["timeframes"]:
                print(f"Fetching stock: {symbol} {timeframe}")
                df = fetcher.fetch_data(symbol, timeframe, 'stocks', start_date)
                results["stocks"][symbol][timeframe] = df
                print(f"  -> Fetched {len(df)} candles")

    return results


if __name__ == "__main__":
    # Test data fetching
    print("Testing data fetcher...")
    fetcher = DataFetcher()

    # Test crypto
    if fetcher.crypto_enabled:
        print("\nFetching BTC/USDT 1h data...")
        df = fetcher.fetch_data("BTC/USDT", "1h", "crypto")
        print(df.head())
        print(f"Total candles: {len(df)}")

    # Test stocks
    if fetcher.stocks_enabled:
        print("\nFetching SPY 1d data...")
        df = fetcher.fetch_data("SPY", "1d", "stocks")
        print(df.head())
        print(f"Total candles: {len(df)}")
