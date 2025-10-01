"""
Run backtests on improved ML models
Compare against buy-and-hold and simple RSI strategy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from src.data.database import TradingDatabase
from src.backtesting.engine import BacktestEngine, SimpleStrategies
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from train_improved import ImprovedFeatureEngineering


def load_data():
    """Load data from database"""
    db = TradingDatabase()

    # Load OHLCV data (timestamp is already the index)
    ohlcv = db.load_ohlcv("SPY", "1h")
    if ohlcv is None or len(ohlcv) == 0:
        raise ValueError("No OHLCV data found")

    # Load indicators (timestamp is already the index)
    indicators = db.load_indicators("SPY", "1h")
    if indicators is not None and len(indicators) > 0:
        # Merge OHLCV with indicators
        df = ohlcv.join(indicators, how='inner')
    else:
        df = ohlcv

    df = df.sort_index()
    return df


def prepare_ml_signals(df, models_dir):
    """Generate ML model signals"""
    print("\n🔧 Preparing ML signals...")

    # Load models
    rf_model = RandomForestModel()
    rf_model.load(models_dir / "random_forest_improved.joblib")

    xgb_model = XGBoostModel()
    xgb_model.load(models_dir / "xgboost_improved.joblib")

    # Create features
    feature_eng = ImprovedFeatureEngineering(prediction_horizon=2)
    features, target = feature_eng.create_all_features(df)
    features = features.fillna(0)

    # Use the exact feature names from the trained model
    model_features = rf_model.feature_names
    print(f"  ✓ Model expects {len(model_features)} features")

    # Filter to only features the model was trained on
    missing_features = [f for f in model_features if f not in features.columns]
    if missing_features:
        print(f"  ⚠ Warning: {len(missing_features)} features missing, filling with zeros")
        for feat in missing_features:
            features[feat] = 0

    # Select features in the same order as training
    X = features[model_features]

    # Align with target
    common_idx = X.index.intersection(target.index)
    X = X.loc[common_idx]

    print(f"  ✓ Created features: {len(X)} samples")

    # Generate predictions
    rf_proba = rf_model.predict_proba(X)
    xgb_proba = xgb_model.predict_proba(X)
    ensemble_proba = (rf_proba + xgb_proba) / 2

    # Convert to signals (1 = long, 0 = cash)
    rf_signals = pd.Series((rf_proba[:, 1] > 0.5).astype(int), index=X.index)
    xgb_signals = pd.Series((xgb_proba[:, 1] > 0.5).astype(int), index=X.index)
    ensemble_signals = pd.Series((ensemble_proba[:, 1] > 0.5).astype(int), index=X.index)

    # Confidence scores
    rf_confidence = pd.Series(np.max(rf_proba, axis=1), index=X.index)
    xgb_confidence = pd.Series(np.max(xgb_proba, axis=1), index=X.index)
    ensemble_confidence = pd.Series(np.max(ensemble_proba, axis=1), index=X.index)

    print("  ✓ Generated ML signals")

    return {
        'Random Forest': rf_signals,
        'XGBoost': xgb_signals,
        'Ensemble': ensemble_signals,
        'Ensemble (60% conf)': ensemble_signals,
    }, {
        'Random Forest': rf_confidence,
        'XGBoost': xgb_confidence,
        'Ensemble': ensemble_confidence,
        'Ensemble (60% conf)': ensemble_confidence,
    }


def run_backtests():
    print("\n" + "="*80)
    print("BACKTESTING ANALYSIS")
    print("="*80)

    # Load data
    print("\n📊 Loading data...")
    df = load_data()
    print(f"  ✓ Loaded {len(df)} candles")

    # Get prices
    prices = df['close']

    # Initialize backtester
    backtester = BacktestEngine(
        initial_capital=10000.0,
        transaction_cost=0.001,  # 0.1%
        slippage=0.0005  # 0.05%
    )

    # Prepare ML signals
    models_dir = Path("models/improved")
    ml_signals, ml_probabilities = prepare_ml_signals(df, models_dir)

    # Prepare baseline strategies
    print("\n📈 Preparing baseline strategies...")
    baseline_signals = {
        'Buy & Hold': SimpleStrategies.buy_and_hold(prices),
    }

    # Add RSI strategy if RSI column exists
    rsi_cols = [col for col in df.columns if 'RSI' in col.upper()]
    if rsi_cols:
        baseline_signals['RSI Strategy'] = SimpleStrategies.rsi_strategy(df, rsi_col=rsi_cols[0])
        print(f"  ✓ RSI strategy using {rsi_cols[0]}")
    else:
        print("  ⚠ No RSI indicator found, skipping RSI strategy")

    print("  ✓ Baseline strategies ready")

    # Combine all strategies
    all_signals = {**baseline_signals, **ml_signals}
    all_probabilities = {**{k: None for k in baseline_signals}, **ml_probabilities}

    # Run backtests
    print("\n" + "="*80)
    print("RUNNING BACKTESTS")
    print("="*80)

    results = {}
    for name, signals in all_signals.items():
        print(f"\n[{name}]")

        # Apply confidence threshold for ensemble
        proba = all_probabilities.get(name)
        threshold = 0.60 if '60% conf' in name else None

        result = backtester.run_backtest(
            prices, signals, proba, threshold
        )

        results[name] = result
        metrics = result['metrics']

        print(f"  Total Return:      {metrics['total_return']:>8.2f}%")
        print(f"  Annualized Return: {metrics['annualized_return']:>8.2f}%")
        print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:>8.2f}")
        print(f"  Max Drawdown:      {metrics['max_drawdown']:>8.2f}%")
        print(f"  Win Rate:          {metrics['win_rate']:>8.2f}%")
        print(f"  Number of Trades:  {metrics['num_trades']:>8.0f}")
        print(f"  Profit Factor:     {metrics['profit_factor']:>8.2f}")

    # Compare strategies
    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)

    comparison_metrics = pd.DataFrame({
        name: result['metrics'] for name, result in results.items()
    }).T

    # Format and display
    display_cols = [
        'total_return', 'annualized_return', 'sharpe_ratio',
        'max_drawdown', 'win_rate', 'num_trades', 'profit_factor'
    ]

    comparison_display = comparison_metrics[display_cols].copy()
    comparison_display = comparison_display.sort_values('sharpe_ratio', ascending=False)

    print("\n" + comparison_display.to_string())

    # Visualize results
    print("\n📊 Creating visualizations...")
    visualize_results(results, prices)

    # Save results
    output_dir = Path("backtests")
    output_dir.mkdir(exist_ok=True)

    comparison_display.to_csv(output_dir / "backtest_results.csv")
    print(f"\n  ✓ Results saved to {output_dir}/backtest_results.csv")

    print("\n" + "="*80)
    print("✅ BACKTESTING COMPLETE!")
    print("="*80)

    return results, comparison_display


def visualize_results(results, prices):
    """Create comprehensive visualization of backtest results"""

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Equity curves
    ax1 = fig.add_subplot(gs[0, :])
    for name, result in results.items():
        equity = result['equity_curve']
        ax1.plot(equity.index, equity, label=name, linewidth=2, alpha=0.8)

    ax1.set_title('Equity Curves ($10,000 Initial Capital)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Returns comparison
    ax2 = fig.add_subplot(gs[1, 0])
    returns = [results[name]['metrics']['total_return'] for name in results.keys()]
    colors = ['green' if r > 0 else 'red' for r in returns]
    ax2.barh(list(results.keys()), returns, color=colors, alpha=0.7)
    ax2.set_xlabel('Total Return (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Total Returns Comparison', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='x')

    # 3. Sharpe ratio comparison
    ax3 = fig.add_subplot(gs[1, 1])
    sharpes = [results[name]['metrics']['sharpe_ratio'] for name in results.keys()]
    colors = ['green' if s > 0 else 'red' for s in sharpes]
    ax3.barh(list(results.keys()), sharpes, color=colors, alpha=0.7)
    ax3.set_xlabel('Sharpe Ratio', fontsize=11, fontweight='bold')
    ax3.set_title('Risk-Adjusted Returns', fontsize=12, fontweight='bold')
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax3.axvline(x=1, color='orange', linestyle=':', linewidth=1, label='Target (1.0)')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='x')

    # 4. Drawdown analysis
    ax4 = fig.add_subplot(gs[2, 0])
    drawdowns = [results[name]['metrics']['max_drawdown'] for name in results.keys()]
    colors = ['red' if d < -20 else 'orange' if d < -10 else 'green' for d in drawdowns]
    ax4.barh(list(results.keys()), drawdowns, color=colors, alpha=0.7)
    ax4.set_xlabel('Maximum Drawdown (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Maximum Drawdown (Lower is Better)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

    # 5. Win rate vs number of trades
    ax5 = fig.add_subplot(gs[2, 1])
    win_rates = [results[name]['metrics']['win_rate'] for name in results.keys()]
    num_trades = [results[name]['metrics']['num_trades'] for name in results.keys()]

    scatter = ax5.scatter(num_trades, win_rates, s=200, alpha=0.6, c=sharpes, cmap='RdYlGn')
    for i, name in enumerate(results.keys()):
        ax5.annotate(name, (num_trades[i], win_rates[i]),
                    fontsize=8, ha='center', va='bottom')

    ax5.set_xlabel('Number of Trades', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
    ax5.set_title('Win Rate vs Trading Frequency', fontsize=12, fontweight='bold')
    ax5.axhline(y=50, color='gray', linestyle='--', linewidth=1, label='50% (Random)')
    ax5.legend(loc='lower right', fontsize=9)
    ax5.grid(True, alpha=0.3)

    plt.colorbar(scatter, ax=ax5, label='Sharpe Ratio')

    fig.suptitle('Backtesting Results - ML Models vs Baselines', fontsize=16, fontweight='bold', y=0.995)

    # Save
    output_dir = Path("backtests")
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / "backtest_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Visualization saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    results, comparison = run_backtests()
