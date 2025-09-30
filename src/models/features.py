"""
Feature engineering module for ML models
Creates features and targets for binary classification
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import TRADING


class FeatureEngineering:
    """Feature engineering for trading ML models"""

    def __init__(self, prediction_horizon: int = None):
        """
        Initialize feature engineering

        Args:
            prediction_horizon: Hours ahead to predict (default from config)
        """
        self.prediction_horizon = prediction_horizon or TRADING["prediction_horizon"]

    def create_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Create binary target: will price be higher in N hours?

        Args:
            df: DataFrame with 'close' column

        Returns:
            Series with binary target (1 = price up, 0 = price down/same)
        """
        # Future price after N periods
        future_price = df['close'].shift(-self.prediction_horizon)
        current_price = df['close']

        # Binary target: 1 if future price is higher, 0 otherwise
        target = (future_price > current_price).astype(int)

        return target

    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-based features

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with price features
        """
        features = pd.DataFrame(index=df.index)

        # Returns over multiple periods
        for period in [1, 2, 4, 8, 24, 168]:  # 1h, 2h, 4h, 8h, 1d, 1w
            features[f'return_{period}h'] = df['close'].pct_change(period)
            features[f'log_return_{period}h'] = np.log(df['close'] / df['close'].shift(period))

        # Price position relative to recent high/low
        for period in [24, 168, 720]:  # 1d, 1w, 1m
            features[f'price_position_{period}h'] = (
                (df['close'] - df['low'].rolling(period).min()) /
                (df['high'].rolling(period).max() - df['low'].rolling(period).min())
            )

        # Distance from moving averages (if they exist)
        for ma_period in [9, 21, 50, 200]:
            if f'EMA_{ma_period}' in df.columns:
                features[f'price_to_ema_{ma_period}'] = (
                    df['close'] / df[f'EMA_{ma_period}'] - 1
                )

        # High-Low range
        features['hl_ratio'] = (df['high'] - df['low']) / df['close']
        features['close_to_high'] = (df['high'] - df['close']) / df['close']
        features['close_to_low'] = (df['close'] - df['low']) / df['close']

        return features

    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volatility features

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with volatility features
        """
        features = pd.DataFrame(index=df.index)

        returns = df['close'].pct_change()

        # Rolling volatility (std of returns)
        for period in [5, 10, 20, 50]:
            features[f'volatility_{period}'] = returns.rolling(period).std()

        # Parkinson volatility (high-low based)
        for period in [10, 20, 50]:
            hl = np.log(df['high'] / df['low'])
            features[f'parkinson_vol_{period}'] = np.sqrt(
                (hl ** 2).rolling(period).mean() / (4 * np.log(2))
            )

        # ATR ratio (if exists)
        if 'atr' in df.columns:
            features['atr_ratio'] = df['atr'] / df['close']

        # Volatility of volatility
        for period in [10, 20]:
            vol = returns.rolling(period).std()
            features[f'vol_of_vol_{period}'] = vol.rolling(period).std()

        return features

    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volume-based features

        Args:
            df: DataFrame with volume data

        Returns:
            DataFrame with volume features
        """
        features = pd.DataFrame(index=df.index)

        # Volume ratios
        for period in [5, 10, 20]:
            vol_ma = df['volume'].rolling(period).mean()
            features[f'volume_ratio_{period}'] = df['volume'] / vol_ma

        # Volume change
        features['volume_change'] = df['volume'].pct_change()

        # Price-volume correlation
        for period in [10, 20, 50]:
            features[f'price_volume_corr_{period}'] = (
                df['close'].rolling(period).corr(df['volume'])
            )

        # Volume momentum
        features['volume_momentum'] = df['volume'] - df['volume'].shift(5)

        return features

    def create_technical_indicator_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and engineer features from technical indicators

        Args:
            df: DataFrame with technical indicators

        Returns:
            DataFrame with indicator features
        """
        features = pd.DataFrame(index=df.index)

        # RSI features
        if 'rsi' in df.columns:
            features['rsi'] = df['rsi']
            features['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            features['rsi_overbought'] = (df['rsi'] > 70).astype(int)
            features['rsi_change'] = df['rsi'].diff()

        # MACD features
        macd_cols = [col for col in df.columns if 'MACD' in col]
        if macd_cols:
            for col in macd_cols:
                features[col] = df[col]

            # MACD crossover
            if any('MACDh' in col for col in macd_cols):
                hist_col = [col for col in macd_cols if 'MACDh' in col][0]
                features['macd_crossover'] = (
                    (df[hist_col] > 0) & (df[hist_col].shift(1) <= 0)
                ).astype(int)
                features['macd_crossunder'] = (
                    (df[hist_col] < 0) & (df[hist_col].shift(1) >= 0)
                ).astype(int)

        # Bollinger Bands features
        bb_cols = [col for col in df.columns if 'BB' in col]
        if bb_cols:
            for col in bb_cols:
                features[col] = df[col]

            # BB position
            if any('BBU' in col for col in bb_cols) and any('BBL' in col for col in bb_cols):
                bbu_col = [col for col in bb_cols if 'BBU' in col][0]
                bbl_col = [col for col in bb_cols if 'BBL' in col][0]
                features['bb_position'] = (
                    (df['close'] - df[bbl_col]) / (df[bbu_col] - df[bbl_col])
                )

        return features

    def create_all_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create all features and target

        Args:
            df: DataFrame with OHLCV data and indicators

        Returns:
            Tuple of (features_df, target_series)
        """
        print("Creating features...")

        feature_dfs = []

        # Price features
        price_features = self.create_price_features(df)
        feature_dfs.append(price_features)
        print(f"  ✓ Price features: {len(price_features.columns)} columns")

        # Volatility features
        vol_features = self.create_volatility_features(df)
        feature_dfs.append(vol_features)
        print(f"  ✓ Volatility features: {len(vol_features.columns)} columns")

        # Volume features
        volume_features = self.create_volume_features(df)
        feature_dfs.append(volume_features)
        print(f"  ✓ Volume features: {len(volume_features.columns)} columns")

        # Technical indicator features
        indicator_features = self.create_technical_indicator_features(df)
        feature_dfs.append(indicator_features)
        print(f"  ✓ Technical indicator features: {len(indicator_features.columns)} columns")

        # Combine all features
        features = pd.concat(feature_dfs, axis=1)

        # Create target
        target = self.create_target(df)
        print(f"  ✓ Target created (prediction horizon: {self.prediction_horizon} periods)")

        # Clean features: replace inf with NaN, then fill/drop
        features = features.replace([np.inf, -np.inf], np.nan)

        # Remove rows with NaN in target (last N rows)
        valid_idx = ~target.isna()
        features = features[valid_idx]
        target = target[valid_idx]

        # Remove rows with too many NaN values in features
        nan_threshold = 0.5  # Drop rows with >50% NaN
        features = features.dropna(thresh=int(len(features.columns) * nan_threshold))
        target = target.loc[features.index]

        # Fill remaining NaN with 0
        features = features.fillna(0)

        print(f"\nTotal features: {len(features.columns)}")
        print(f"Total samples: {len(features)} (after cleaning)")
        print(f"Target distribution: {target.value_counts().to_dict()}")

        return features, target

    def prepare_data_for_training(self, features: pd.DataFrame, target: pd.Series,
                                  train_size: float = 0.8) -> dict:
        """
        Prepare data for training with walk-forward split

        Args:
            features: Feature DataFrame
            target: Target Series
            train_size: Proportion for training (default 0.8)

        Returns:
            Dictionary with train/test splits
        """
        # Remove NaN values
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        features = features[valid_idx]
        target = target[valid_idx]

        # Walk-forward split (no shuffling - maintain time order)
        split_idx = int(len(features) * train_size)

        X_train = features.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        y_train = target.iloc[:split_idx]
        y_test = target.iloc[split_idx:]

        print(f"\nData split:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Testing: {len(X_test)} samples")
        print(f"  Train target distribution: {y_train.value_counts().to_dict()}")
        print(f"  Test target distribution: {y_test.value_counts().to_dict()}")

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': list(features.columns)
        }


if __name__ == "__main__":
    # Test feature engineering
    from src.data import DataPipeline

    print("Testing feature engineering...")
    pipeline = DataPipeline()

    # Get data with indicators
    df = pipeline.get_data_with_indicators('SPY', '1h')
    print(f"\nLoaded {len(df)} candles")

    # Create features
    fe = FeatureEngineering(prediction_horizon=4)
    features, target = fe.create_all_features(df)

    print("\nFeature sample:")
    print(features.tail())

    # Prepare for training
    data = fe.prepare_data_for_training(features, target)

    pipeline.close()
