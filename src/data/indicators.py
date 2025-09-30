"""
Technical indicators module using pandas-ta
Calculates RSI, MACD, Bollinger Bands, EMA, ATR, and volume indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import INDICATORS

try:
    import pandas_ta as ta
except ImportError:
    ta = None
    print("Warning: pandas-ta not installed. Install with: pip install pandas-ta")


class TechnicalIndicators:
    """Calculate technical indicators for trading data"""

    def __init__(self, config: dict = INDICATORS):
        """
        Initialize with indicator configuration

        Args:
            config: Dictionary with indicator settings from config
        """
        self.config = config

    def calculate_rsi(self, df: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)

        Args:
            df: DataFrame with 'close' column
            period: RSI period (default from config)

        Returns:
            Series with RSI values
        """
        if not self.config["rsi"]["enabled"]:
            return pd.Series(index=df.index, dtype=float)

        period = period or self.config["rsi"]["period"]

        if ta is not None:
            return ta.rsi(df['close'], length=period)
        else:
            # Manual RSI calculation if pandas-ta not available
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence)

        Args:
            df: DataFrame with 'close' column

        Returns:
            DataFrame with columns: macd, macd_signal, macd_histogram
        """
        if not self.config["macd"]["enabled"]:
            return pd.DataFrame(index=df.index)

        fast = self.config["macd"]["fast"]
        slow = self.config["macd"]["slow"]
        signal = self.config["macd"]["signal"]

        if ta is not None:
            macd = ta.macd(df['close'], fast=fast, slow=slow, signal=signal)
            if macd is not None:
                return macd

        # Manual MACD calculation
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame({
            f'MACD_{fast}_{slow}_{signal}': macd_line,
            f'MACDs_{fast}_{slow}_{signal}': signal_line,
            f'MACDh_{fast}_{slow}_{signal}': histogram
        })

    def calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands

        Args:
            df: DataFrame with 'close' column

        Returns:
            DataFrame with columns: bb_lower, bb_middle, bb_upper, bb_bandwidth
        """
        if not self.config["bollinger_bands"]["enabled"]:
            return pd.DataFrame(index=df.index)

        period = self.config["bollinger_bands"]["period"]
        std = self.config["bollinger_bands"]["std"]

        if ta is not None:
            bb = ta.bbands(df['close'], length=period, std=std)
            if bb is not None:
                return bb

        # Manual Bollinger Bands calculation
        sma = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()

        upper = sma + (rolling_std * std)
        lower = sma - (rolling_std * std)
        bandwidth = (upper - lower) / sma

        return pd.DataFrame({
            f'BBL_{period}_{std}': lower,
            f'BBM_{period}_{std}': sma,
            f'BBU_{period}_{std}': upper,
            f'BBB_{period}_{std}': bandwidth
        })

    def calculate_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Exponential Moving Averages

        Args:
            df: DataFrame with 'close' column

        Returns:
            DataFrame with EMA columns for each configured period
        """
        if not self.config["ema"]["enabled"]:
            return pd.DataFrame(index=df.index)

        periods = self.config["ema"]["periods"]
        ema_df = pd.DataFrame(index=df.index)

        for period in periods:
            if ta is not None:
                ema_df[f'EMA_{period}'] = ta.ema(df['close'], length=period)
            else:
                ema_df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        return ema_df

    def calculate_atr(self, df: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
        """
        Calculate Average True Range (ATR)

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: ATR period (default from config)

        Returns:
            Series with ATR values
        """
        if not self.config["atr"]["enabled"]:
            return pd.Series(index=df.index, dtype=float)

        period = period or self.config["atr"]["period"]

        if ta is not None:
            return ta.atr(df['high'], df['low'], df['close'], length=period)

        # Manual ATR calculation
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based indicators

        Args:
            df: DataFrame with 'volume' column

        Returns:
            DataFrame with volume indicators
        """
        if not self.config["volume"]["enabled"]:
            return pd.DataFrame(index=df.index)

        sma_period = self.config["volume"]["sma_period"]

        volume_df = pd.DataFrame(index=df.index)

        # Volume SMA
        volume_df[f'volume_sma_{sma_period}'] = df['volume'].rolling(window=sma_period).mean()

        # Volume ratio (current volume / average volume)
        volume_df['volume_ratio'] = df['volume'] / volume_df[f'volume_sma_{sma_period}']

        # On-Balance Volume (OBV)
        if ta is not None:
            volume_df['obv'] = ta.obv(df['close'], df['volume'])
        else:
            obv = [0]
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv.append(obv[-1] + df['volume'].iloc[i])
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv.append(obv[-1] - df['volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            volume_df['obv'] = obv

        return volume_df

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all enabled indicators

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all indicator columns added
        """
        if df.empty:
            return df

        result_df = df.copy()

        # RSI
        if self.config["rsi"]["enabled"]:
            result_df['rsi'] = self.calculate_rsi(df)

        # MACD
        if self.config["macd"]["enabled"]:
            macd_df = self.calculate_macd(df)
            result_df = pd.concat([result_df, macd_df], axis=1)

        # Bollinger Bands
        if self.config["bollinger_bands"]["enabled"]:
            bb_df = self.calculate_bollinger_bands(df)
            result_df = pd.concat([result_df, bb_df], axis=1)

        # EMA
        if self.config["ema"]["enabled"]:
            ema_df = self.calculate_ema(df)
            result_df = pd.concat([result_df, ema_df], axis=1)

        # ATR
        if self.config["atr"]["enabled"]:
            result_df['atr'] = self.calculate_atr(df)

        # Volume indicators
        if self.config["volume"]["enabled"]:
            volume_df = self.calculate_volume_indicators(df)
            result_df = pd.concat([result_df, volume_df], axis=1)

        # Add price change features
        result_df['returns'] = df['close'].pct_change()
        result_df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        result_df['price_change'] = df['close'].diff()

        # Add momentum features
        result_df['momentum_1'] = df['close'] - df['close'].shift(1)
        result_df['momentum_3'] = df['close'] - df['close'].shift(3)
        result_df['momentum_5'] = df['close'] - df['close'].shift(5)

        # Add volatility features
        result_df['volatility_10'] = df['close'].pct_change().rolling(window=10).std()
        result_df['volatility_30'] = df['close'].pct_change().rolling(window=30).std()

        return result_df

    def get_indicator_columns(self, df: pd.DataFrame) -> list:
        """
        Get list of indicator column names (excludes OHLCV)

        Args:
            df: DataFrame with indicators

        Returns:
            List of indicator column names
        """
        base_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        return [col for col in df.columns if col not in base_columns]


def add_indicators_to_data(df: pd.DataFrame, config: dict = INDICATORS) -> pd.DataFrame:
    """
    Convenience function to add indicators to OHLCV data

    Args:
        df: DataFrame with OHLCV data
        config: Indicator configuration

    Returns:
        DataFrame with indicators added
    """
    indicators = TechnicalIndicators(config)
    return indicators.calculate_all_indicators(df)


if __name__ == "__main__":
    # Test indicators
    print("Testing technical indicators...")

    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })
    sample_data.set_index('timestamp', inplace=True)

    # Calculate indicators
    indicators = TechnicalIndicators()
    result = indicators.calculate_all_indicators(sample_data)

    print("\nSample data with indicators:")
    print(result.tail())
    print(f"\nTotal columns: {len(result.columns)}")
    print(f"Indicator columns: {indicators.get_indicator_columns(result)}")
