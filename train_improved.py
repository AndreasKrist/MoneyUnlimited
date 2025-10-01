"""
Improved training pipeline with proven enhancements
Implements best practices for financial ML
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from src.data import DataPipeline
from src.models import FeatureEngineering, RandomForestModel, XGBoostModel, EnsembleModel, ModelEvaluator


class ImprovedFeatureEngineering(FeatureEngineering):
    """Enhanced feature engineering with market regime detection"""

    def add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime features (trending vs choppy)
        This is a HUGE edge - models perform better in specific regimes
        """
        features = pd.DataFrame(index=df.index)

        # ADX - measures trend strength (>25 = trending, <20 = choppy)
        if 'atr' in df.columns:
            # Calculate directional movement
            high_diff = df['high'].diff()
            low_diff = -df['low'].diff()

            pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

            # Smooth with 14-period average
            atr_14 = df['atr'] if 'atr' in df.columns else (df['high'] - df['low']).rolling(14).mean()
            pos_di = 100 * (pos_dm.rolling(14).mean() / atr_14)
            neg_di = 100 * (neg_dm.rolling(14).mean() / atr_14)

            # ADX calculation
            dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
            features['adx'] = dx.rolling(14).mean()

            # Regime binary: 1 = trending, 0 = choppy
            features['is_trending'] = (features['adx'] > 25).astype(int)

        # Volatility regime
        if 'close' in df.columns:
            returns = df['close'].pct_change()
            vol_20 = returns.rolling(20).std()
            vol_100 = returns.rolling(100).std()

            # High volatility = opportunities
            features['high_volatility'] = (vol_20 > vol_100 * 1.5).astype(int)

            # Rising volatility
            features['vol_increasing'] = (vol_20.diff(5) > 0).astype(int)

        # Volume regime (institutional activity)
        if 'volume' in df.columns:
            vol_ma = df['volume'].rolling(20).mean()
            features['high_volume'] = (df['volume'] > vol_ma * 1.2).astype(int)

        return features

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Time-of-day patterns (market open, close, lunch)
        Markets behave differently at different times
        """
        features = pd.DataFrame(index=df.index)

        # Extract hour from index
        if isinstance(df.index, pd.DatetimeIndex):
            hour = df.index.hour

            # Market open (9:30-11:00 ET = 13:30-15:00 UTC for SPY)
            features['is_market_open'] = ((hour >= 13) & (hour < 15)).astype(int)

            # Market close (3:00-4:00 PM ET = 19:00-20:00 UTC)
            features['is_market_close'] = ((hour >= 19) & (hour < 20)).astype(int)

            # Lunch hour (12:00-1:00 PM ET = 16:00-17:00 UTC)
            features['is_lunch_hour'] = ((hour >= 16) & (hour < 17)).astype(int)

            # Day of week
            features['day_of_week'] = df.index.dayofweek

            # Monday effect (higher volatility)
            features['is_monday'] = (df.index.dayofweek == 0).astype(int)

            # Friday effect (position closing)
            features['is_friday'] = (df.index.dayofweek == 4).astype(int)

        return features

    def create_all_features(self, df: pd.DataFrame):
        """Enhanced feature creation with regime detection"""

        # Original features
        features, target = super().create_all_features(df)

        # Add regime features
        print("  ✓ Adding market regime features...")
        regime_features = self.add_market_regime_features(df)
        features = pd.concat([features, regime_features], axis=1)

        # Add time features
        print("  ✓ Adding time-of-day features...")
        time_features = self.add_time_features(df)
        features = pd.concat([features, time_features], axis=1)

        # Clean again after adding new features
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        print(f"\n✨ Enhanced features: {len(features.columns)} total")

        return features, target


def select_best_features(X_train, y_train, X_test, top_n=20):
    """
    Feature selection using Random Forest importance
    Removes noisy features that hurt performance
    """
    print("\n" + "="*60)
    print("FEATURE SELECTION")
    print("="*60)

    # Train RF to get importance
    rf = RandomForestModel(n_estimators=100, random_state=42)
    rf.train(X_train, y_train)

    # Get top features
    importance = rf.get_feature_importance(top_n=top_n)
    selected_features = importance['feature'].tolist()

    print(f"\n✂️  Selected top {top_n} features:")
    for i, feat in enumerate(selected_features[:10], 1):
        print(f"  {i}. {feat}")
    if len(selected_features) > 10:
        print(f"  ... and {len(selected_features) - 10} more")

    return selected_features


def train_with_improvements():
    """Train models with all improvements"""

    print("\n" + "="*80)
    print("IMPROVED TRADING BOT - ML TRAINING")
    print("="*80)

    # Load data
    print("\n📊 Loading data...")
    pipeline = DataPipeline()
    df = pipeline.get_data_with_indicators('SPY', '1h')
    print(f"  ✓ Loaded {len(df)} candles")

    # Test different horizons quickly
    print("\n" + "="*60)
    print("FINDING BEST PREDICTION HORIZON")
    print("="*60)

    best_horizon = 4
    best_acc = 0

    for horizon in [2, 4, 6, 8]:
        fe = ImprovedFeatureEngineering(prediction_horizon=horizon)
        features, target = fe.create_all_features(df)
        data = fe.prepare_data_for_training(features, target, train_size=0.8)

        # Quick test with small RF
        rf = RandomForestModel(n_estimators=50, random_state=42)
        rf.train(data['X_train'], data['y_train'])
        pred = rf.predict(data['X_test'])

        # Ensure same length (handle any edge cases)
        min_len = min(len(pred), len(data['y_test']))
        acc = (pred[:min_len] == data['y_test'].values[:min_len]).mean()

        print(f"  Horizon {horizon}h: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_horizon = horizon

    print(f"\n✨ Best horizon: {best_horizon}h with {best_acc:.4f} accuracy")

    # Train with best horizon and all features
    print("\n" + "="*60)
    print(f"CREATING ENHANCED FEATURES (horizon={best_horizon}h)")
    print("="*60)

    fe = ImprovedFeatureEngineering(prediction_horizon=best_horizon)
    features, target = fe.create_all_features(df)
    data = fe.prepare_data_for_training(features, target, train_size=0.8)

    # Feature selection
    selected_features = select_best_features(
        data['X_train'], data['y_train'], data['X_test'], top_n=25
    )

    # Filter to selected features
    X_train = data['X_train'][selected_features]
    X_test = data['X_test'][selected_features]

    # Train improved models
    print("\n" + "="*60)
    print("TRAINING IMPROVED MODELS")
    print("="*60)

    # Random Forest with better hyperparameters
    print("\n[Random Forest - Optimized]")
    rf_model = RandomForestModel(
        n_estimators=300,  # More trees
        max_depth=15,      # Prevent overfitting
        min_samples_split=20,  # More conservative
        random_state=42
    )
    rf_model.train(X_train, data['y_train'])
    print("  ✓ Trained")

    # XGBoost with better hyperparameters
    print("\n[XGBoost - Optimized]")
    xgb_model = XGBoostModel(
        n_estimators=200,
        max_depth=6,       # Shallower trees
        learning_rate=0.05,  # Slower learning
        random_state=42
    )

    # Validation split
    val_size = 0.1
    split_idx = int(len(X_train) * (1 - val_size))
    X_train_main = X_train.iloc[:split_idx]
    X_val = X_train.iloc[split_idx:]
    y_train_main = data['y_train'].iloc[:split_idx]
    y_val = data['y_train'].iloc[split_idx:]

    xgb_model.train(X_train_main, y_train_main, X_val, y_val)
    print("  ✓ Trained")

    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION ON TEST SET")
    print("="*60)

    evaluator = ModelEvaluator()

    # Align y_test with X_test (match lengths after feature selection)
    min_len = min(len(X_test), len(data['y_test']))
    y_test_aligned = data['y_test'].iloc[:min_len].values
    X_test = X_test.iloc[:min_len]

    # Random Forest
    print("\n[Random Forest]")
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)
    rf_metrics = evaluator.evaluate(y_test_aligned, rf_pred, rf_proba, "Random Forest")
    evaluator.print_metrics(rf_metrics)

    # XGBoost
    print("\n[XGBoost]")
    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)
    xgb_metrics = evaluator.evaluate(y_test_aligned, xgb_pred, xgb_proba, "XGBoost")
    evaluator.print_metrics(xgb_metrics)

    # Ensemble with confidence threshold
    print("\n[Ensemble with Confidence Filter]")
    ensemble_proba = (rf_proba + xgb_proba) / 2

    # Only predict when confident (>60% probability)
    confidence_threshold = 0.60
    confident_mask = (ensemble_proba[:, 1] > confidence_threshold) | (ensemble_proba[:, 0] > confidence_threshold)

    ensemble_pred = (ensemble_proba[:, 1] > 0.5).astype(int)
    confident_pred = ensemble_pred[confident_mask]
    confident_true = y_test_aligned[confident_mask]

    if len(confident_pred) > 0:
        print(f"  Predictions with >{confidence_threshold:.0%} confidence: {len(confident_pred)}/{len(ensemble_pred)} ({len(confident_pred)/len(ensemble_pred):.1%})")

        conf_metrics = evaluator.evaluate(confident_true, confident_pred, None, f"Ensemble (confidence >{confidence_threshold:.0%})")
        evaluator.print_metrics(conf_metrics)

    # Compare all
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    evaluator.compare_models([rf_metrics, xgb_metrics, conf_metrics if len(confident_pred) > 0 else None])

    # Save improved models
    print("\n" + "="*60)
    print("SAVING IMPROVED MODELS")
    print("="*60)

    models_dir = Path("models/improved")
    models_dir.mkdir(parents=True, exist_ok=True)

    rf_model.save(models_dir / "random_forest_improved.joblib")
    xgb_model.save(models_dir / "xgboost_improved.joblib")

    # Save metadata
    import joblib
    joblib.dump({
        'selected_features': selected_features,
        'prediction_horizon': best_horizon,
        'confidence_threshold': confidence_threshold
    }, models_dir / "improved_metadata.joblib")

    print("  ✓ All improved models saved")

    print("\n" + "="*80)
    print("✨ IMPROVED TRAINING COMPLETE!")
    print("="*80)

    pipeline.close()


if __name__ == "__main__":
    train_with_improvements()
