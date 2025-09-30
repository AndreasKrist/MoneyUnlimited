"""
XGBoost classifier for trading predictions
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Optional
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import MODELS_DIR


class XGBoostModel:
    """XGBoost classifier wrapper"""

    def __init__(self, n_estimators: int = 200, max_depth: int = 8,
                 learning_rate: float = 0.1, random_state: int = 42):
        """
        Initialize XGBoost model

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            random_state: Random seed
        """
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            eval_metric='logloss',
            use_label_encoder=False,
            scale_pos_weight=1  # Can adjust for class imbalance
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> dict:
        """
        Train the model

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Optional validation features
            y_val: Optional validation target

        Returns:
            Dictionary with training info
        """
        print("\nTraining XGBoost...")

        # Store feature names
        self.feature_names = list(X_train.columns)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)

        # Prepare validation set if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            eval_set = [(X_scaled, y_train), (X_val_scaled, y_val)]

        # Train model
        self.model.fit(
            X_scaled, y_train,
            eval_set=eval_set,
            verbose=False
        )
        self.is_fitted = True

        # Training accuracy
        train_pred = self.model.predict(X_scaled)
        train_acc = (train_pred == y_train).mean()

        print(f"  ✓ Training accuracy: {train_acc:.4f}")

        result = {
            'train_accuracy': train_acc,
            'n_features': len(self.feature_names),
            'n_samples': len(X_train)
        }

        if eval_set:
            val_pred = self.model.predict(X_val_scaled)
            val_acc = (val_pred == y_val).mean()
            print(f"  ✓ Validation accuracy: {val_acc:.4f}")
            result['val_accuracy'] = val_acc

        return result

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Features

        Returns:
            Predicted classes
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities

        Args:
            X: Features

        Returns:
            Predicted probabilities [prob_class_0, prob_class_1]
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet")

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df.head(top_n)

    def save(self, filepath: Optional[Path] = None):
        """Save model to disk"""
        if filepath is None:
            filepath = MODELS_DIR / "xgboost_model.joblib"

        filepath.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }, filepath)

        print(f"  ✓ Model saved to {filepath}")

    def load(self, filepath: Optional[Path] = None):
        """Load model from disk"""
        if filepath is None:
            filepath = MODELS_DIR / "xgboost_model.joblib"

        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.is_fitted = data['is_fitted']

        print(f"  ✓ Model loaded from {filepath}")
