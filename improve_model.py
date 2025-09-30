"""
Model improvement strategies
Identifies ways to boost trading bot performance
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data import DataPipeline
from src.models import FeatureEngineering, RandomForestModel
import pandas as pd
import numpy as np


def analyze_feature_importance():
    """Find which features actually matter"""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)

    # Load data
    pipeline = DataPipeline()
    df = pipeline.get_data_with_indicators('SPY', '1h')

    # Create features
    fe = FeatureEngineering(prediction_horizon=4)
    features, target = fe.create_all_features(df)

    # Train model
    data = fe.prepare_data_for_training(features, target, train_size=0.8)

    rf = RandomForestModel(n_estimators=100)
    rf.train(data['X_train'], data['y_train'])

    # Get top features
    importance = rf.get_feature_importance(top_n=20)

    print("\n🎯 TOP 20 MOST IMPORTANT FEATURES:")
    print("="*60)
    for i, row in importance.iterrows():
        print(f"{i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")

    # Identify low-value features
    all_importance = rf.get_feature_importance(top_n=100)
    low_value = all_importance[all_importance['importance'] < 0.01]

    print(f"\n❌ LOW-VALUE FEATURES (importance < 0.01): {len(low_value)}")
    print("These can probably be removed:")
    for _, row in low_value.head(10).iterrows():
        print(f"  - {row['feature']:<30} {row['importance']:.4f}")

    pipeline.close()
    return importance


def test_different_horizons():
    """Test which prediction horizon works best"""
    print("\n" + "="*60)
    print("PREDICTION HORIZON OPTIMIZATION")
    print("="*60)

    pipeline = DataPipeline()
    df = pipeline.get_data_with_indicators('SPY', '1h')

    results = []

    for horizon in [1, 2, 4, 6, 8, 12, 24]:
        print(f"\nTesting horizon = {horizon} periods...")

        fe = FeatureEngineering(prediction_horizon=horizon)
        features, target = fe.create_all_features(df)
        data = fe.prepare_data_for_training(features, target, train_size=0.8)

        # Quick train with fewer trees
        rf = RandomForestModel(n_estimators=50)
        rf.train(data['X_train'], data['y_train'])

        # Test accuracy
        pred = rf.predict(data['X_test'])
        accuracy = (pred == data['y_test'].values).mean()

        results.append({
            'horizon': horizon,
            'accuracy': accuracy,
            'samples': len(data['y_test'])
        })

        print(f"  Accuracy: {accuracy:.4f}")

    pipeline.close()

    # Show best
    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df['accuracy'].idxmax()]

    print("\n" + "="*60)
    print(f"🏆 BEST HORIZON: {best['horizon']} periods")
    print(f"   Accuracy: {best['accuracy']:.4f}")
    print("="*60)

    return results_df


def suggest_improvements():
    """Suggest specific improvements"""
    print("\n" + "="*80)
    print("💡 IMPROVEMENT SUGGESTIONS")
    print("="*80)

    suggestions = [
        {
            "strategy": "1. Use Top Features Only",
            "description": "Train with only top 15-20 features (reduces noise)",
            "expected_gain": "+2-5% accuracy",
            "implementation": "Filter features by importance > 0.02"
        },
        {
            "strategy": "2. Optimize Prediction Horizon",
            "description": "Test different time horizons (1h, 2h, 4h, 8h)",
            "expected_gain": "+1-3% accuracy",
            "implementation": "Run test_different_horizons() above"
        },
        {
            "strategy": "3. Add Market Regime Filter",
            "description": "Only trade in trending markets (skip choppy periods)",
            "expected_gain": "+3-7% accuracy",
            "implementation": "Check if ADX > 25 or ATR increasing"
        },
        {
            "strategy": "4. Tune Hyperparameters",
            "description": "Optimize n_estimators, max_depth, learning_rate",
            "expected_gain": "+1-2% accuracy",
            "implementation": "Use GridSearchCV or RandomizedSearchCV"
        },
        {
            "strategy": "5. Ensemble with Probability Threshold",
            "description": "Only predict when confidence > 60%",
            "expected_gain": "Higher precision, fewer trades",
            "implementation": "Filter predictions where proba > 0.6"
        },
        {
            "strategy": "6. Add Time-of-Day Features",
            "description": "Market open, close, lunch hour patterns",
            "expected_gain": "+1-2% accuracy",
            "implementation": "Extract hour from timestamp"
        },
        {
            "strategy": "7. Use Class Weights",
            "description": "Penalize false positives more (reduce losses)",
            "expected_gain": "Better risk/reward",
            "implementation": "Adjust class_weight in models"
        },
        {
            "strategy": "8. Add More Data",
            "description": "Train on 5+ years instead of 2 years",
            "expected_gain": "+2-4% accuracy",
            "implementation": "Change lookback_days in config"
        }
    ]

    for i, s in enumerate(suggestions, 1):
        print(f"\n{i}. {s['strategy']}")
        print(f"   {s['description']}")
        print(f"   Expected: {s['expected_gain']}")
        print(f"   How: {s['implementation']}")


def quick_improvement_test():
    """Test top features only (quick win)"""
    print("\n" + "="*60)
    print("QUICK TEST: Top Features Only")
    print("="*60)

    pipeline = DataPipeline()
    df = pipeline.get_data_with_indicators('SPY', '1h')

    fe = FeatureEngineering(prediction_horizon=4)
    features, target = fe.create_all_features(df)

    # Train with all features
    data = fe.prepare_data_for_training(features, target, train_size=0.8)

    print("\nBaseline (all 55 features):")
    rf_all = RandomForestModel(n_estimators=100)
    rf_all.train(data['X_train'], data['y_train'])
    pred_all = rf_all.predict(data['X_test'])
    acc_all = (pred_all == data['y_test'].values).mean()
    print(f"  Accuracy: {acc_all:.4f}")

    # Get top features
    importance = rf_all.get_feature_importance(top_n=20)
    top_features = importance['feature'].tolist()

    # Train with top 20 features only
    print(f"\nWith top 20 features only:")
    X_train_top = data['X_train'][top_features]
    X_test_top = data['X_test'][top_features]

    rf_top = RandomForestModel(n_estimators=100)
    rf_top.train(X_train_top, data['y_train'])
    pred_top = rf_top.predict(X_test_top)
    acc_top = (pred_top == data['y_test'].values).mean()
    print(f"  Accuracy: {acc_top:.4f}")

    diff = (acc_top - acc_all) * 100
    print(f"\n{'🚀 IMPROVEMENT' if diff > 0 else '📉 WORSE'}: {diff:+.2f}%")

    pipeline.close()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TRADING BOT - MODEL IMPROVEMENT ANALYSIS")
    print("="*80)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', action='store_true', help='Analyze feature importance')
    parser.add_argument('--horizons', action='store_true', help='Test different prediction horizons')
    parser.add_argument('--quick', action='store_true', help='Quick test with top features')
    parser.add_argument('--all', action='store_true', help='Run all analyses')

    args = parser.parse_args()

    if args.all or args.features:
        analyze_feature_importance()

    if args.all or args.horizons:
        test_different_horizons()

    if args.all or args.quick:
        quick_improvement_test()

    if not any([args.features, args.horizons, args.quick, args.all]):
        # Just show suggestions
        suggest_improvements()
