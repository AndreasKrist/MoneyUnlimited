"""
Fetch data for all symbols used in the dashboard
"""
from src.data.pipeline import DataPipeline
from config.config import DATA_SOURCES

def fetch_all_data():
    """Fetch data for all dashboard symbols"""
    pipeline = DataPipeline()

    # Stocks to fetch
    stocks = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA"]
    timeframes = ["1h", "1d"]

    print("="*80)
    print("FETCHING STOCK DATA")
    print("="*80)

    for symbol in stocks:
        print(f"\n📊 Fetching {symbol}...")
        for timeframe in timeframes:
            try:
                df = pipeline.update_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    source='stocks',
                    force_refresh=False
                )
                if df is not None and len(df) > 0:
                    print(f"  ✓ {timeframe}: {len(df)} candles")
                else:
                    print(f"  ✗ {timeframe}: No data returned")
            except Exception as e:
                print(f"  ✗ {timeframe}: {str(e)}")

    # Try crypto if ccxt is available
    print("\n" + "="*80)
    print("FETCHING CRYPTO DATA")
    print("="*80)

    if DATA_SOURCES["crypto"]["enabled"]:
        cryptos = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]

        for symbol in cryptos:
            print(f"\n₿ Fetching {symbol}...")
            for timeframe in timeframes:
                try:
                    df = pipeline.update_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        source='crypto',
                        force_refresh=False
                    )
                    if df is not None and len(df) > 0:
                        print(f"  ✓ {timeframe}: {len(df)} candles")
                    else:
                        print(f"  ✗ {timeframe}: No data returned")
                except Exception as e:
                    print(f"  ✗ {timeframe}: {str(e)}")
    else:
        print("\n⚠ Crypto fetching disabled (ccxt not installed)")
        print("To enable: pip install ccxt --user")
        print("Then set crypto.enabled = True in config/config.py")

    print("\n" + "="*80)
    print("✅ DATA FETCHING COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    fetch_all_data()
