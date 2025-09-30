"""
Database module for caching OHLCV data and indicators
Uses SQLite for local storage
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import DB_PATH


class TradingDatabase:
    """Manages SQLite database for trading data storage"""

    def __init__(self, db_path: Path = DB_PATH):
        """Initialize database connection and create tables if needed"""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def _create_tables(self):
        """Create necessary tables if they don't exist"""

        # OHLCV data table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                source TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, timestamp)
            )
        """)

        # Technical indicators table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                indicator_name TEXT NOT NULL,
                indicator_value REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, timestamp, indicator_name)
            )
        """)

        # Create indexes for faster queries
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe
            ON ohlcv_data(symbol, timeframe, timestamp)
        """)

        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_indicators_symbol_timeframe
            ON indicators(symbol, timeframe, timestamp)
        """)

        self.conn.commit()

    def save_ohlcv(self, df: pd.DataFrame, symbol: str, timeframe: str, source: str):
        """
        Save OHLCV data to database

        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume]
            symbol: Trading pair symbol (e.g., 'BTC/USDT', 'SPY')
            timeframe: Candle timeframe (e.g., '1h', '4h', '1d')
            source: Data source ('ccxt' or 'yfinance')
        """
        df = df.copy()
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        df['source'] = source

        # Ensure timestamp is in Unix format (seconds)
        if not pd.api.types.is_integer_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**9

        # Insert or replace data
        df[['symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'source']].to_sql(
            'ohlcv_data',
            self.conn,
            if_exists='append',
            index=False,
            method='multi'
        )

        # Remove duplicates (keep latest)
        self.cursor.execute("""
            DELETE FROM ohlcv_data
            WHERE id NOT IN (
                SELECT MAX(id)
                FROM ohlcv_data
                GROUP BY symbol, timeframe, timestamp
            )
        """)

        self.conn.commit()

    def load_ohlcv(self, symbol: str, timeframe: str,
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load OHLCV data from database

        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            DataFrame with OHLCV data
        """
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_data
            WHERE symbol = ? AND timeframe = ?
        """
        params = [symbol, timeframe]

        if start_date:
            query += " AND timestamp >= ?"
            params.append(int(start_date.timestamp()))

        if end_date:
            query += " AND timestamp <= ?"
            params.append(int(end_date.timestamp()))

        query += " ORDER BY timestamp ASC"

        df = pd.read_sql_query(query, self.conn, params=params)

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)

        return df

    def save_indicators(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """
        Save technical indicators to database

        Args:
            df: DataFrame with timestamp index and indicator columns
            symbol: Trading pair symbol
            timeframe: Candle timeframe
        """
        if df.empty:
            return

        # Prepare data for insertion
        records = []
        for timestamp, row in df.iterrows():
            ts = int(timestamp.timestamp()) if isinstance(timestamp, pd.Timestamp) else int(timestamp)

            for col in df.columns:
                if pd.notna(row[col]):  # Skip NaN values
                    records.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'timestamp': ts,
                        'indicator_name': col,
                        'indicator_value': float(row[col])
                    })

        if records:
            indicators_df = pd.DataFrame(records)
            indicators_df.to_sql(
                'indicators',
                self.conn,
                if_exists='append',
                index=False,
                method='multi'
            )

            # Remove duplicates
            self.cursor.execute("""
                DELETE FROM indicators
                WHERE id NOT IN (
                    SELECT MAX(id)
                    FROM indicators
                    GROUP BY symbol, timeframe, timestamp, indicator_name
                )
            """)

            self.conn.commit()

    def load_indicators(self, symbol: str, timeframe: str,
                       indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load technical indicators from database

        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            indicators: Optional list of specific indicators to load

        Returns:
            DataFrame with indicators as columns
        """
        query = """
            SELECT timestamp, indicator_name, indicator_value
            FROM indicators
            WHERE symbol = ? AND timeframe = ?
        """
        params = [symbol, timeframe]

        if indicators:
            placeholders = ','.join(['?'] * len(indicators))
            query += f" AND indicator_name IN ({placeholders})"
            params.extend(indicators)

        query += " ORDER BY timestamp ASC"

        df = pd.read_sql_query(query, self.conn, params=params)

        if not df.empty:
            # Pivot to get indicators as columns
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.pivot(index='timestamp', columns='indicator_name', values='indicator_value')

        return df

    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """Get the most recent timestamp for a symbol/timeframe"""
        query = """
            SELECT MAX(timestamp) as max_ts
            FROM ohlcv_data
            WHERE symbol = ? AND timeframe = ?
        """
        result = pd.read_sql_query(query, self.conn, params=[symbol, timeframe])

        if result['max_ts'].iloc[0]:
            return datetime.fromtimestamp(result['max_ts'].iloc[0])
        return None

    def clear_data(self, symbol: Optional[str] = None, timeframe: Optional[str] = None):
        """Clear data from database (use with caution)"""
        if symbol and timeframe:
            self.cursor.execute("DELETE FROM ohlcv_data WHERE symbol = ? AND timeframe = ?",
                              (symbol, timeframe))
            self.cursor.execute("DELETE FROM indicators WHERE symbol = ? AND timeframe = ?",
                              (symbol, timeframe))
        elif symbol:
            self.cursor.execute("DELETE FROM ohlcv_data WHERE symbol = ?", (symbol,))
            self.cursor.execute("DELETE FROM indicators WHERE symbol = ?", (symbol,))
        else:
            self.cursor.execute("DELETE FROM ohlcv_data")
            self.cursor.execute("DELETE FROM indicators")

        self.conn.commit()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
