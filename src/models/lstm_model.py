"""
LSTM Neural Network for trading predictions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Optional, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import MODELS_DIR

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not installed. LSTM model disabled.")
    tf = None
    keras = None
    TF_AVAILABLE = False


class LSTMModel:
    """LSTM Neural Network wrapper"""

    def __init__(self, sequence_length: int = 24, lstm_units: int = 64,
                 dropout: float = 0.2, random_state: int = 42):
        """
        Initialize LSTM model

        Args:
            sequence_length: Number of time steps to look back
            lstm_units: Number of LSTM units
            dropout: Dropout rate
            random_state: Random seed
        """
        if tf is None:
            raise RuntimeError("TensorFlow not installed")

        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.random_state = random_state

        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False

    def _build_model(self, n_features: int):
        """Build LSTM architecture"""
        model = keras.Sequential([
            # First LSTM layer
            layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                input_shape=(self.sequence_length, n_features)
            ),
            layers.Dropout(self.dropout),

            # Second LSTM layer
            layers.LSTM(self.lstm_units // 2, return_sequences=False),
            layers.Dropout(self.dropout),

            # Dense layers
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.dropout),
            layers.Dense(16, activation='relu'),

            # Output layer (binary classification)
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )

        return model

    def _create_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None
                         ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences for LSTM

        Args:
            X: Feature array
            y: Optional target array

        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_sequences = []
        y_sequences = [] if y is not None else None

        for i in range(len(X) - self.sequence_length):
            X_sequences.append(X[i:i + self.sequence_length])
            if y is not None:
                y_sequences.append(y[i + self.sequence_length])

        X_sequences = np.array(X_sequences)
        if y is not None:
            y_sequences = np.array(y_sequences)

        return X_sequences, y_sequences

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None,
              epochs: int = 50, batch_size: int = 32) -> dict:
        """
        Train the model

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Optional validation features
            y_val: Optional validation target
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Dictionary with training info
        """
        print("\nTraining LSTM...")

        # Store feature names
        self.feature_names = list(X_train.columns)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_train.values)

        print(f"  Sequence shape: {X_seq.shape}")
        print(f"  Samples after sequencing: {len(X_seq)}")

        # Build model
        self.model = self._build_model(n_features=X_train.shape[1])

        # Prepare callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                verbose=0
            )
        ]

        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val.values)
            validation_data = (X_val_seq, y_val_seq)

        # Train model
        history = self.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0
        )

        self.is_fitted = True

        # Training accuracy
        train_loss, train_acc, train_auc = self.model.evaluate(X_seq, y_seq, verbose=0)
        print(f"  ✓ Training accuracy: {train_acc:.4f}")
        print(f"  ✓ Training AUC: {train_auc:.4f}")

        result = {
            'train_accuracy': train_acc,
            'train_auc': train_auc,
            'train_loss': train_loss,
            'n_features': len(self.feature_names),
            'n_samples': len(X_seq),
            'epochs_trained': len(history.history['loss'])
        }

        if validation_data:
            val_loss, val_acc, val_auc = self.model.evaluate(X_val_seq, y_val_seq, verbose=0)
            print(f"  ✓ Validation accuracy: {val_acc:.4f}")
            print(f"  ✓ Validation AUC: {val_auc:.4f}")
            result['val_accuracy'] = val_acc
            result['val_auc'] = val_auc
            result['val_loss'] = val_loss

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
        X_seq, _ = self._create_sequences(X_scaled)

        predictions = self.model.predict(X_seq, verbose=0)
        predictions = (predictions > 0.5).astype(int).flatten()

        # Pad predictions to match input length
        # (first sequence_length predictions are unknown)
        padding = np.full(self.sequence_length, -1)
        return np.concatenate([padding, predictions])

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
        X_seq, _ = self._create_sequences(X_scaled)

        predictions = self.model.predict(X_seq, verbose=0)

        # Convert to [prob_0, prob_1] format
        prob_1 = predictions.flatten()
        prob_0 = 1 - prob_1
        proba = np.column_stack([prob_0, prob_1])

        # Pad to match input length
        padding = np.full((self.sequence_length, 2), 0.5)
        return np.vstack([padding, proba])

    def save(self, filepath: Optional[Path] = None):
        """Save model to disk"""
        if filepath is None:
            filepath = MODELS_DIR / "lstm_model"

        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save Keras model
        self.model.save(str(filepath) + ".keras")

        # Save scaler and metadata
        import joblib
        joblib.dump({
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'sequence_length': self.sequence_length,
            'is_fitted': self.is_fitted
        }, str(filepath) + "_metadata.joblib")

        print(f"  ✓ Model saved to {filepath}")

    def load(self, filepath: Optional[Path] = None):
        """Load model from disk"""
        if filepath is None:
            filepath = MODELS_DIR / "lstm_model"

        # Load Keras model
        self.model = keras.models.load_model(str(filepath) + ".keras")

        # Load scaler and metadata
        import joblib
        data = joblib.load(str(filepath) + "_metadata.joblib")
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.sequence_length = data['sequence_length']
        self.is_fitted = data['is_fitted']

        print(f"  ✓ Model loaded from {filepath}")
