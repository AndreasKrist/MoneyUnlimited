"""
Quick test script for Phase 2 ML models
Tests feature engineering and model training
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np


def test_feature_engineering():
    """Test feature engineering"""
    print("\n" + "="*60)
    print("TEST 1: Feature Engineering")
    print("="*60)

    from src.data import DataPipeline
    from src.models import FeatureEngineering

    pipeline = DataPipeline()
    df = pipeline.get_data_with_indicators('SPY', '1h')
    print(f"✓ Loaded {len(df)} candles")

    fe = FeatureEngineering(prediction_horizon=4)
    features, target = fe.create_all_features(df)

    print(f"\n✓ Features shape: {features.shape}")
    print(f"✓ Target shape: {target.shape}")
    print(f"✓ Target distribution: {target.value_counts().to_dict()}")

    pipeline.close()


def test_individual_models():
    """Test individual model training"""
    print("\n" + "="*60)
    print("TEST 2: Individual Model Training (Small Sample)")
    print("="*60)

    from src.data import DataPipeline
    from src.models import FeatureEngineering, RandomForestModel, XGBoostModel

    # Load data
    pipeline = DataPipeline()
    df = pipeline.get_data_with_indicators('SPY', '1h')

    # Use only last 500 samples for quick test
    df = df.tail(500)
    print(f"Using {len(df)} samples for testing")

    # Create features
    fe = FeatureEngineering(prediction_horizon=4)
    features, target = fe.create_all_features(df)

    # Split data
    data = fe.prepare_data_for_training(features, target, train_size=0.8)

    # Test Random Forest
    print("\n[Random Forest]")
    rf = RandomForestModel(n_estimators=50)  # Fewer trees for speed
    rf.train(data['X_train'], data['y_train'])

    # Test predictions
    pred = rf.predict(data['X_test'])
    accuracy = (pred == data['y_test'].values).mean()
    print(f"  ✓ Test accuracy: {accuracy:.4f}")

    # Feature importance
    importance = rf.get_feature_importance(top_n=5)
    print(f"\n  Top 5 features:")
    for _, row in importance.iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")

    # Test XGBoost
    print("\n[XGBoost]")
    xgb = XGBoostModel(n_estimators=50)
    xgb.train(data['X_train'], data['y_train'])

    pred = xgb.predict(data['X_test'])
    accuracy = (pred == data['y_test'].values).mean()
    print(f"  ✓ Test accuracy: {accuracy:.4f}")

    pipeline.close()


def test_evaluation():
    """Test evaluation metrics"""
    print("\n" + "="*60)
    print("TEST 3: Model Evaluation")
    print("="*60)

    from src.models import ModelEvaluator

    # Generate dummy predictions for testing
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_proba = np.random.rand(100, 2)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, y_proba, "Test Model")

    print(f"✓ Accuracy: {metrics['accuracy']:.4f}")
    print(f"✓ Precision: {metrics['precision']:.4f}")
    print(f"✓ Recall: {metrics['recall']:.4f}")
    print(f"✓ F1 Score: {metrics['f1_score']:.4f}")
    print(f"✓ ROC AUC: {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print(" PHASE 2 ML MODELS - TEST SUITE")
    print("="*80)

    try:
        test_feature_engineering()
        test_individual_models()
        test_evaluation()

        print("\n" + "="*80)
        print(" ALL TESTS COMPLETED")
        print("="*80)
        print("\n✓ Phase 2 ML infrastructure is working!")
        print("\nTo train full models, run:")
        print("  python src/models/train_pipeline.py")
        print("\nOptions:")
        print("  --symbol SPY      # Trading symbol")
        print("  --timeframe 1h    # Data timeframe")
        print("  --horizon 4       # Prediction horizon")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
