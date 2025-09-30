"""
Main training pipeline for ML models
Orchestrates feature engineering, model training, and evaluation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data import DataPipeline
from src.models.features import FeatureEngineering
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel
from src.models.ensemble import EnsembleModel
from src.models.evaluation import ModelEvaluator
from config.config import TRADING, MODELS


class TrainingPipeline:
    """Main training pipeline"""

    def __init__(self, symbol: str = "SPY", timeframe: str = "1h",
                 prediction_horizon: int = 4):
        """
        Initialize training pipeline

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            prediction_horizon: Hours ahead to predict
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.prediction_horizon = prediction_horizon

        self.data_pipeline = DataPipeline()
        self.feature_eng = FeatureEngineering(prediction_horizon)
        self.evaluator = ModelEvaluator()

        self.data = None
        self.features = None
        self.target = None
        self.data_split = None

    def load_data(self):
        """Load data from database"""
        print(f"\n{'='*60}")
        print(f"LOADING DATA: {self.symbol} {self.timeframe}")
        print(f"{'='*60}")

        self.data = self.data_pipeline.get_data_with_indicators(
            self.symbol, self.timeframe
        )

        print(f"Loaded {len(self.data)} candles")
        print(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")

        return self.data

    def create_features(self):
        """Create features and target"""
        print(f"\n{'='*60}")
        print("FEATURE ENGINEERING")
        print(f"{'='*60}")

        self.features, self.target = self.feature_eng.create_all_features(self.data)

        return self.features, self.target

    def split_data(self, train_size: float = 0.8):
        """Split data for training"""
        print(f"\n{'='*60}")
        print("DATA SPLITTING")
        print(f"{'='*60}")

        self.data_split = self.feature_eng.prepare_data_for_training(
            self.features, self.target, train_size
        )

        return self.data_split

    def train_individual_models(self):
        """Train individual models"""
        print(f"\n{'='*60}")
        print("TRAINING INDIVIDUAL MODELS")
        print(f"{'='*60}")

        results = {}

        # Create validation split (10% of training data)
        val_size = 0.1
        split_idx = int(len(self.data_split['X_train']) * (1 - val_size))

        X_train_main = self.data_split['X_train'].iloc[:split_idx]
        X_val = self.data_split['X_train'].iloc[split_idx:]
        y_train_main = self.data_split['y_train'].iloc[:split_idx]
        y_val = self.data_split['y_train'].iloc[split_idx:]

        # Train Random Forest
        print("\n[1/3] Random Forest")
        rf_model = RandomForestModel()
        results['random_forest'] = rf_model.train(
            self.data_split['X_train'],
            self.data_split['y_train']
        )
        results['random_forest']['model'] = rf_model

        # Train XGBoost
        print("\n[2/3] XGBoost")
        xgb_model = XGBoostModel()
        results['xgboost'] = xgb_model.train(
            X_train_main, y_train_main,
            X_val, y_val
        )
        results['xgboost']['model'] = xgb_model

        # Train LSTM
        print("\n[3/3] LSTM")
        try:
            lstm_model = LSTMModel(sequence_length=24)
            results['lstm'] = lstm_model.train(
                X_train_main, y_train_main,
                X_val, y_val,
                epochs=30,
                batch_size=32
            )
            results['lstm']['model'] = lstm_model
        except Exception as e:
            print(f"  ✗ LSTM training failed: {e}")
            results['lstm'] = {'error': str(e)}

        return results

    def train_ensemble(self):
        """Train ensemble model"""
        print(f"\n{'='*60}")
        print("TRAINING ENSEMBLE MODEL")
        print(f"{'='*60}")

        # Create ensemble
        self.ensemble = EnsembleModel()

        # Validation split
        val_size = 0.1
        split_idx = int(len(self.data_split['X_train']) * (1 - val_size))

        X_train_main = self.data_split['X_train'].iloc[:split_idx]
        X_val = self.data_split['X_train'].iloc[split_idx:]
        y_train_main = self.data_split['y_train'].iloc[:split_idx]
        y_val = self.data_split['y_train'].iloc[split_idx:]

        # Train
        results = self.ensemble.train(X_train_main, y_train_main, X_val, y_val)

        return self.ensemble, results

    def evaluate_models(self):
        """Evaluate all models on test set"""
        print(f"\n{'='*60}")
        print("MODEL EVALUATION ON TEST SET")
        print(f"{'='*60}")

        X_test = self.data_split['X_test']
        y_test = self.data_split['y_test']

        results = {}

        # Evaluate ensemble
        print("\n[ENSEMBLE]")
        ensemble_pred = self.ensemble.predict(X_test)
        ensemble_proba = self.ensemble.predict_proba(X_test)
        results['ensemble'] = self.evaluator.evaluate(
            y_test.values, ensemble_pred, ensemble_proba, "Ensemble"
        )
        self.evaluator.print_metrics(results['ensemble'])

        # Evaluate individual models
        individual_preds = self.ensemble.get_individual_predictions(X_test)

        for name, pred in individual_preds.items():
            if pred is not None:
                print(f"\n[{name.upper()}]")
                results[name] = self.evaluator.evaluate(
                    y_test.values, pred, None, name.capitalize()
                )
                self.evaluator.print_metrics(results[name])

        # Compare models
        metrics_list = [v for v in results.values() if v is not None]
        self.evaluator.compare_models(metrics_list)

        return results

    def save_models(self):
        """Save trained models"""
        print(f"\n{'='*60}")
        print("SAVING MODELS")
        print(f"{'='*60}")

        self.ensemble.save()
        print("✓ All models saved")

    def run_full_pipeline(self):
        """Run complete training pipeline"""
        print("\n" + "="*80)
        print("TRADING BOT - ML TRAINING PIPELINE")
        print(f"Symbol: {self.symbol} | Timeframe: {self.timeframe}")
        print(f"Prediction Horizon: {self.prediction_horizon} periods")
        print("="*80)

        try:
            # Load data
            self.load_data()

            # Create features
            self.create_features()

            # Split data
            self.split_data()

            # Train ensemble
            self.train_ensemble()

            # Evaluate
            results = self.evaluate_models()

            # Save models
            self.save_models()

            print("\n" + "="*80)
            print("✓ TRAINING PIPELINE COMPLETE!")
            print("="*80)

            return results

        except Exception as e:
            print(f"\n✗ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None

        finally:
            self.data_pipeline.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train ML models for trading")
    parser.add_argument('--symbol', type=str, default='SPY',
                       help='Trading symbol (default: SPY)')
    parser.add_argument('--timeframe', type=str, default='1h',
                       help='Data timeframe (default: 1h)')
    parser.add_argument('--horizon', type=int, default=4,
                       help='Prediction horizon in periods (default: 4)')

    args = parser.parse_args()

    # Run pipeline
    pipeline = TrainingPipeline(
        symbol=args.symbol,
        timeframe=args.timeframe,
        prediction_horizon=args.horizon
    )

    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
