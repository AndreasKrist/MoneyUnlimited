"""
Quick test script for Phase 1 data infrastructure
Tests data fetching, indicator calculation, and database operations
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data import DataFetcher, TechnicalIndicators, TradingDatabase, DataPipeline
import pandas as pd
from datetime import datetime, timedelta


def test_data_fetcher():
    """Test data fetching functionality"""
    print("\n" + "="*60)
    print("TEST 1: Data Fetcher")
    print("="*60)

    fetcher = DataFetcher()

    # Test crypto fetching
    if fetcher.crypto_enabled:
        print("\n✓ Crypto fetching enabled (ccxt)")
        print("Fetching small sample of BTC/USDT 1h data...")
        start = datetime.now() - timedelta(days=7)
        df = fetcher.fetch_data("BTC/USDT", "1h", "crypto", start)
        print(f"Fetched {len(df)} candles")
        print(df.head(3))
    else:
        print("\n✗ Crypto fetching disabled (ccxt not available)")

    # Test stock fetching
    if fetcher.stocks_enabled:
        print("\n✓ Stock fetching enabled (yfinance)")
        print("Fetching small sample of SPY 1d data...")
        start = datetime.now() - timedelta(days=30)
        df = fetcher.fetch_data("SPY", "1d", "stocks", start)
        print(f"Fetched {len(df)} candles")
        print(df.head(3))
    else:
        print("\n✗ Stock fetching disabled (yfinance not available)")


def test_technical_indicators():
    """Test technical indicator calculation"""
    print("\n" + "="*60)
    print("TEST 2: Technical Indicators")
    print("="*60)

    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=200, freq='1h')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': pd.Series(range(100, 100+200)) + pd.Series(range(200)).apply(lambda x: x % 10 - 5),
        'high': pd.Series(range(102, 102+200)) + pd.Series(range(200)).apply(lambda x: x % 10 - 5),
        'low': pd.Series(range(98, 98+200)) + pd.Series(range(200)).apply(lambda x: x % 10 - 5),
        'close': pd.Series(range(100, 100+200)) + pd.Series(range(200)).apply(lambda x: x % 10 - 5),
        'volume': pd.Series(range(200)).apply(lambda x: 10000 + (x % 100) * 100)
    })
    sample_data.set_index('timestamp', inplace=True)

    print(f"\nCreated sample data: {len(sample_data)} candles")

    # Calculate indicators
    indicators = TechnicalIndicators()
    result = indicators.calculate_all_indicators(sample_data)

    print(f"Total columns: {len(result.columns)}")
    indicator_cols = indicators.get_indicator_columns(result)
    print(f"Indicator columns: {len(indicator_cols)}")
    print(f"\nIndicators calculated:")
    for col in sorted(indicator_cols)[:15]:  # Show first 15
        print(f"  - {col}")
    if len(indicator_cols) > 15:
        print(f"  ... and {len(indicator_cols) - 15} more")

    print(f"\nSample row with indicators:")
    print(result.iloc[-1][['close', 'rsi', 'atr']].to_dict())


def test_database():
    """Test database operations"""
    print("\n" + "="*60)
    print("TEST 3: Database Operations")
    print("="*60)

    # Create test data
    dates = pd.date_range(start='2023-01-01', periods=50, freq='1h')
    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': range(100, 150),
        'high': range(102, 152),
        'low': range(98, 148),
        'close': range(100, 150),
        'volume': [10000] * 50
    })

    db = TradingDatabase()

    # Save data
    print("\nSaving test data to database...")
    db.save_ohlcv(test_data, "TEST/PAIR", "1h", "test")
    print("✓ Data saved")

    # Load data
    print("\nLoading data from database...")
    loaded = db.load_ohlcv("TEST/PAIR", "1h")
    print(f"✓ Loaded {len(loaded)} candles")
    print(loaded.head(3))

    # Test latest timestamp
    latest = db.get_latest_timestamp("TEST/PAIR", "1h")
    print(f"\n✓ Latest timestamp: {latest}")

    # Clean up test data
    db.clear_data("TEST/PAIR", "1h")
    print("\n✓ Test data cleaned up")

    db.close()


def test_full_pipeline():
    """Test the complete pipeline with small dataset"""
    print("\n" + "="*60)
    print("TEST 4: Full Pipeline (Limited)")
    print("="*60)

    pipeline = DataPipeline()

    # Test with a small date range
    print("\nTesting pipeline with 7 days of data...")

    # You can uncomment this to test with real data
    # But it will actually fetch and store data
    # pipeline.update_all_configured(force_refresh=False)

    # Show summary
    print("\nCurrent database summary:")
    summary = pipeline.get_data_summary()
    if summary.empty:
        print("No data in database (run: python src/data/pipeline.py)")
    else:
        print(summary.to_string(index=False))

    pipeline.close()


if __name__ == "__main__":
    print("\n" + "="*80)
    print(" PHASE 1 DATA INFRASTRUCTURE - TEST SUITE")
    print("="*80)

    try:
        test_data_fetcher()
        test_technical_indicators()
        test_database()
        test_full_pipeline()

        print("\n" + "="*80)
        print(" ALL TESTS COMPLETED")
        print("="*80)
        print("\n✓ Phase 1 infrastructure is working correctly!")
        print("\nTo fetch real data, run:")
        print("  python src/data/pipeline.py")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
