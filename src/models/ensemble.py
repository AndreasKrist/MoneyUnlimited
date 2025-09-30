"""
Ensemble model combining Random Forest, XGBoost, and LSTM
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel

# Try to import LSTM (optional if TensorFlow not installed)
LSTM_AVAILABLE = False
try:
    from src.models import lstm_model
    if lstm_model.TF_AVAILABLE:
        from src.models.lstm_model import LSTMModel
        LSTM_AVAILABLE = True
except Exception:
    pass

from config.config import MODELS_DIR


class EnsembleModel:
    """Ensemble model with weighted voting"""

    def __init__(self, models: Optional[Dict[str, any]] = None,
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble

        Args:
            models: Dictionary of models {name: model_instance}
            weights: Dictionary of weights {name: weight}
        """
        if models is None:
            # Default: create all available models
            self.models = {
                'random_forest': RandomForestModel(),
                'xgboost': XGBoostModel(),
            }

            # Add LSTM only if TensorFlow is available
            if LSTM_AVAILABLE:
                self.models['lstm'] = LSTMModel()
            else:
                print("Note: Ensemble will use Random Forest + XGBoost only (LSTM unavailable)")
        else:
            self.models = models

        # Default equal weights
        if weights is None:
            self.weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
        else:
            # Normalize weights
            total = sum(weights.values())
            self.weights = {k: v / total for k, v in weights.items()}

        self.is_fitted = False

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> dict:
        """
        Train all models in ensemble

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Optional validation features
            y_val: Optional validation target

        Returns:
            Dictionary with training info for each model
        """
        print("\n" + "="*60)
        print("TRAINING ENSEMBLE MODELS")
        print("="*60)

        results = {}

        # Train each model
        for name, model in self.models.items():
            print(f"\n[{name.upper()}]")
            try:
                if name == 'lstm':
                    # LSTM needs validation data
                    result = model.train(X_train, y_train, X_val, y_val)
                elif name in ['xgboost']:
                    # XGBoost can use validation data
                    result = model.train(X_train, y_train, X_val, y_val)
                else:
                    # Random Forest
                    result = model.train(X_train, y_train)

                results[name] = result
                print(f"  ✓ {name} trained successfully")

            except Exception as e:
                print(f"  ✗ Error training {name}: {e}")
                results[name] = {'error': str(e)}

        self.is_fitted = True

        print("\n" + "="*60)
        print("ENSEMBLE TRAINING COMPLETE")
        print("="*60)

        return results

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions using weighted voting

        Args:
            X: Features

        Returns:
            Predicted classes
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not trained yet")

        predictions = []
        valid_weights = []

        for name, model in self.models.items():
            try:
                pred = model.predict(X)

                # Handle LSTM padding (predictions start after sequence_length)
                if name == 'lstm' and hasattr(model, 'sequence_length'):
                    # Skip padded values (-1)
                    valid_mask = pred != -1
                    if not valid_mask.all():
                        # For padded values, use neutral prediction
                        pred = np.where(valid_mask, pred, 0.5)

                predictions.append(pred)
                valid_weights.append(self.weights[name])

            except Exception as e:
                print(f"Warning: {name} prediction failed: {e}")

        if not predictions:
            raise RuntimeError("All models failed to predict")

        # Normalize weights for valid models
        valid_weights = np.array(valid_weights)
        valid_weights = valid_weights / valid_weights.sum()

        # Weighted voting
        predictions = np.array(predictions)
        weighted_pred = np.average(predictions, axis=0, weights=valid_weights)

        # Convert to binary (threshold = 0.5)
        return (weighted_pred > 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities using weighted averaging

        Args:
            X: Features

        Returns:
            Predicted probabilities [prob_class_0, prob_class_1]
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not trained yet")

        probabilities = []
        valid_weights = []

        for name, model in self.models.items():
            try:
                proba = model.predict_proba(X)

                # Ensure 2D array [N, 2]
                if proba.ndim == 1:
                    proba = np.column_stack([1 - proba, proba])

                probabilities.append(proba)
                valid_weights.append(self.weights[name])

            except Exception as e:
                print(f"Warning: {name} probability prediction failed: {e}")

        if not probabilities:
            raise RuntimeError("All models failed to predict probabilities")

        # Normalize weights
        valid_weights = np.array(valid_weights)
        valid_weights = valid_weights / valid_weights.sum()

        # Weighted average of probabilities
        probabilities = np.array(probabilities)  # Shape: [n_models, n_samples, 2]
        weighted_proba = np.average(probabilities, axis=0, weights=valid_weights)

        return weighted_proba

    def get_individual_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get predictions from each individual model

        Args:
            X: Features

        Returns:
            Dictionary of predictions {model_name: predictions}
        """
        predictions = {}

        for name, model in self.models.items():
            try:
                predictions[name] = model.predict(X)
            except Exception as e:
                print(f"Warning: {name} prediction failed: {e}")
                predictions[name] = None

        return predictions

    def save(self, directory: Optional[Path] = None):
        """Save all models in ensemble"""
        if directory is None:
            directory = MODELS_DIR / "ensemble"

        directory.mkdir(parents=True, exist_ok=True)

        print("\nSaving ensemble models...")

        for name, model in self.models.items():
            try:
                model.save(directory / f"{name}_model")
                print(f"  ✓ {name} saved")
            except Exception as e:
                print(f"  ✗ Error saving {name}: {e}")

        # Save weights
        import joblib
        joblib.dump({
            'weights': self.weights,
            'is_fitted': self.is_fitted
        }, directory / "ensemble_metadata.joblib")

        print(f"  ✓ Ensemble metadata saved")

    def load(self, directory: Optional[Path] = None):
        """Load all models in ensemble"""
        if directory is None:
            directory = MODELS_DIR / "ensemble"

        print("\nLoading ensemble models...")

        for name, model in self.models.items():
            try:
                model.load(directory / f"{name}_model")
                print(f"  ✓ {name} loaded")
            except Exception as e:
                print(f"  ✗ Error loading {name}: {e}")

        # Load weights
        import joblib
        data = joblib.load(directory / "ensemble_metadata.joblib")
        self.weights = data['weights']
        self.is_fitted = data['is_fitted']

        print(f"  ✓ Ensemble metadata loaded")

    def set_weights(self, weights: Dict[str, float]):
        """
        Update model weights

        Args:
            weights: Dictionary of new weights
        """
        # Normalize
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}
        print(f"Updated weights: {self.weights}")
