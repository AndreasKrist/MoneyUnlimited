# 🚀 Quick Start Guide

## Get Started in 3 Steps

### 1️⃣ Install Dependencies
```powershell
pip install -r requirements.txt --user
```

### 2️⃣ Fetch Data (2-3 minutes)
```powershell
python src/data/pipeline.py
```
This downloads 2 years of SPY stock data with technical indicators.

### 3️⃣ Train ML Models (3-5 minutes)
```powershell
python src/models/train_pipeline.py
```
This trains Random Forest, XGBoost, and LSTM models.

## ✅ What You Get

After these steps, you'll have:
- **3,984 candles** of SPY data with 25 indicators
- **3 trained ML models** (RF, XGBoost, LSTM)
- **Ensemble predictor** combining all models
- **Performance metrics** (accuracy, precision, recall, F1, ROC AUC)

## 📊 Expected Results

Training output shows:
```
EVALUATION: Ensemble
============================================================
Accuracy:    0.5500-0.6500  (55-65%)
Precision:   0.5200-0.6200
Recall:      0.5000-0.6500
F1 Score:    0.5100-0.6300
ROC AUC:     0.5800-0.7000
```

**Note**: 55-65% accuracy is EXCELLENT for financial prediction!
Anything better than 50% (random) is valuable.

## 🎯 Quick Tests

```powershell
# Test Phase 1 (Data Pipeline)
python test_phase1.py

# Test Phase 2 (ML Models)
python test_phase2.py

# View data summary
python src/data/pipeline.py --summary
```

## 🔧 Common Issues

**Issue**: `ccxt not installed`
**Fix**: `pip install ccxt --user`

**Issue**: `TensorFlow not available`
**Fix**: `pip install tensorflow --user`

**Issue**: `Too many SQL variables`
**Fix**: Already handled in database.py with batching

## 🎮 Next Steps

Once trained, you can:

### Make Predictions
```python
from src.models import EnsembleModel

ensemble = EnsembleModel()
ensemble.load()

# On new data
predictions = ensemble.predict(features_df)
print(f"Prediction: {'UP' if predictions[-1] == 1 else 'DOWN'}")
```

### Analyze Feature Importance
```python
from src.models import RandomForestModel

rf = RandomForestModel()
rf.load()
importance = rf.get_feature_importance(top_n=10)
print(importance)
```

### Compare Models
Results are automatically compared during training!

## 📁 Project Files

```
d:\MoneyUnlimited/
├── data/
│   └── trading_data.db        # Your data (auto-created)
├── models/
│   └── ensemble/              # Trained models (auto-created)
├── src/
│   ├── data/                  # Phase 1: Data pipeline
│   └── models/                # Phase 2: ML models
├── config/
│   └── config.py              # Settings (edit if needed)
└── requirements.txt           # Dependencies
```

## 🎓 Learn More

- **Phase 1 Details**: See [README.md](README.md)
- **Phase 2 Details**: See [PHASE2_INFO.md](PHASE2_INFO.md)
- **Configuration**: Edit [config/config.py](config/config.py)

## 🤔 FAQ

**Q: Can I use crypto data?**
A: Yes! First install ccxt, then enable crypto in config.py

**Q: How do I change the prediction horizon?**
A: Use `--horizon N` flag: `python src/models/train_pipeline.py --horizon 8`

**Q: Can I train on other stocks?**
A: Yes! `python src/models/train_pipeline.py --symbol AAPL --timeframe 1h`

**Q: Why is accuracy only 55-60%?**
A: Financial markets are extremely difficult to predict!
   55-60% is actually excellent and potentially profitable.

**Q: Is this ready for real trading?**
A: NO! This is for learning/portfolio purposes only.
   Real trading needs Phase 3 (backtesting) and proper risk management.

## 🎉 Success!

If everything works, you now have a working ML trading bot!
Ready for **Phase 3** (Backtesting) or **Phase 4** (Dashboard)?
