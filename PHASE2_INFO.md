# Phase 2: ML Models - COMPLETED ✅

## 🤖 Features Built

### Feature Engineering ([features.py](src/models/features.py))
- **60+ engineered features** from price, volume, volatility, and indicators
- Returns over multiple timeframes: 1h, 2h, 4h, 8h, 1d, 1w
- Price position relative to recent highs/lows
- Volatility features: rolling std, Parkinson volatility, volatility-of-volatility
- Volume features: ratios, momentum, price-volume correlation
- Technical indicator transformations (RSI zones, MACD crossovers, BB position)

### ML Models

1. **Random Forest Classifier** ([random_forest_model.py](src/models/random_forest_model.py))
   - 200 trees with balanced class weights
   - Feature importance analysis
   - Handles class imbalance automatically

2. **XGBoost Classifier** ([xgboost_model.py](src/models/xgboost_model.py))
   - Gradient boosting with early stopping
   - Optional validation monitoring
   - Feature importance rankings

3. **LSTM Neural Network** ([lstm_model.py](src/models/lstm_model.py))
   - 2-layer LSTM with dropout (0.2)
   - Sequence length: 24 periods (lookback)
   - Early stopping & learning rate reduction
   - Uses TensorFlow/Keras

### Ensemble System ([ensemble.py](src/models/ensemble.py))
- Weighted voting from all 3 models
- Soft probability averaging
- Individual model predictions available
- Configurable model weights

### Evaluation ([evaluation.py](src/models/evaluation.py))
- **Metrics**: Accuracy, Precision, Recall, F1 Score, ROC AUC
- Confusion matrix visualization
- Model comparison reports
- ROC curve plotting

### Walk-Forward Validation
- **No data leakage** - maintains time-series order
- 80/20 train/test split (no shuffling)
- 10% validation split from training data
- Proper handling of LSTM sequences

## 🚀 Usage

### Train All Models
```bash
python src/models/train_pipeline.py
```

### Train with Custom Parameters
```bash
python src/models/train_pipeline.py --symbol SPY --timeframe 1h --horizon 4
```

### Test Phase 2
```bash
python test_phase2.py
```

### Use in Python
```python
from src.models import TrainingPipeline, EnsembleModel

# Option 1: Run full training pipeline
pipeline = TrainingPipeline(symbol='SPY', timeframe='1h', prediction_horizon=4)
results = pipeline.run_full_pipeline()

# Option 2: Load trained models
ensemble = EnsembleModel()
ensemble.load()

# Make predictions
predictions = ensemble.predict(features_df)
probabilities = ensemble.predict_proba(features_df)

# Get individual model predictions
individual_preds = ensemble.get_individual_predictions(features_df)
```

## 📊 Expected Output

Training will show:
1. Data loading summary
2. Feature engineering stats (60+ features created)
3. Train/test split info
4. Training progress for each model:
   - Random Forest: Training accuracy
   - XGBoost: Train & validation accuracy
   - LSTM: Train & validation accuracy + AUC
5. Test set evaluation:
   - Accuracy, Precision, Recall, F1
   - Confusion matrix
   - Model comparison table

## 🎯 Target Metrics

- **Accuracy**: > 55%
- **Precision**: > 0.55
- **F1 Score**: > 0.50
- **ROC AUC**: > 0.60

These are realistic targets for financial time-series prediction. Anything significantly better than 50% (random guessing) is valuable!

## 📁 Models Saved

After training, models are saved to:
```
models/
├── ensemble/
│   ├── random_forest_model.joblib
│   ├── xgboost_model.joblib
│   ├── lstm_model.keras
│   ├── lstm_model_metadata.joblib
│   └── ensemble_metadata.joblib
```

## 🔍 What Gets Predicted?

**Binary Classification**: Will the price be **higher** in N periods (default: 4 hours)?
- **Class 1** (UP): Price will increase
- **Class 0** (DOWN): Price will decrease or stay same

## 🧪 Testing

Run tests to verify:
```bash
python test_phase2.py
```

Tests include:
1. Feature engineering (60+ features)
2. Individual model training (RF, XGBoost)
3. Evaluation metrics calculation

## ⚡ Performance Tips

- **Full training**: ~3-5 minutes on SPY 1h data (3,484 candles)
- **Quick test**: Use `--horizon 1` for faster predictions
- **Memory**: LSTM uses ~500MB RAM during training
- **CPU**: Use all cores (n_jobs=-1 in RF/XGBoost)

## 🐛 Troubleshooting

**LSTM fails to train:**
- Ensure TensorFlow is installed: `pip install tensorflow`
- Check you have enough data (>500 candles recommended)

**Out of memory:**
- Reduce LSTM sequence_length (default: 24)
- Reduce batch_size (default: 32)
- Train on smaller timeframe subset

**Poor accuracy:**
- Normal for financial data! 55-60% is good
- Try different prediction horizons (--horizon 1 or 8)
- Adjust train/test split ratio
- Check class imbalance in target

## 🎓 Next Steps

With trained models, you can:
1. ✅ Make predictions on new data
2. ✅ Evaluate performance metrics
3. ✅ Compare model strategies
4. 🔜 **Phase 3**: Backtest with realistic trading simulation
5. 🔜 **Phase 4**: Build live dashboard
