"""
Main data pipeline orchestrator
Fetches data, calculates indicators, and stores in database
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import DATA_SOURCES, HISTORICAL_DATA
from src.data.fetcher import DataFetcher
from src.data.indicators import TechnicalIndicators
from src.data.database import TradingDatabase


class DataPipeline:
    """Orchestrates the complete data pipeline"""

    def __init__(self):
        """Initialize pipeline components"""
        self.fetcher = DataFetcher()
        self.indicators = TechnicalIndicators()
        self.db = TradingDatabase()

    def update_data(self, symbol: str, timeframe: str, source: str,
                   force_refresh: bool = False) -> pd.DataFrame:
        """
        Update data for a specific symbol and timeframe

        Args:
            symbol: Trading pair or ticker
            timeframe: Candle timeframe
            source: Data source ('crypto' or 'stocks')
            force_refresh: If True, fetch all data from scratch

        Returns:
            DataFrame with OHLCV data and indicators
        """
        print(f"\n{'='*60}")
        print(f"Updating: {symbol} {timeframe} ({source})")
        print(f"{'='*60}")

        # Determine start date
        if force_refresh:
            start_date = None
            print("Force refresh enabled - fetching all historical data")
        else:
            # Check for existing data
            latest_ts = self.db.get_latest_timestamp(symbol, timeframe)
            if latest_ts:
                # Fetch from last timestamp
                start_date = latest_ts
                print(f"Existing data found. Fetching from: {start_date}")
            else:
                start_date = None
                print("No existing data. Fetching full history")

        # Fetch new data
        print(f"Fetching data...")
        df = self.fetcher.fetch_data(symbol, timeframe, source, start_date)

        if df.empty:
            print("No new data fetched")
            # Try loading existing data
            df = self.db.load_ohlcv(symbol, timeframe)
            if df.empty:
                print("No existing data in database")
                return df
        else:
            print(f"Fetched {len(df)} new candles")

            # Save OHLCV data
            print("Saving OHLCV data to database...")
            self.db.save_ohlcv(df, symbol, timeframe, source)

        # Load complete dataset
        df = self.db.load_ohlcv(symbol, timeframe)
        print(f"Total candles in database: {len(df)}")

        # Calculate indicators
        print("Calculating technical indicators...")
        df_with_indicators = self.indicators.calculate_all_indicators(df)

        # Extract only indicator columns for storage
        indicator_cols = self.indicators.get_indicator_columns(df_with_indicators)
        indicators_df = df_with_indicators[indicator_cols]

        # Save indicators
        print("Saving indicators to database...")
        self.db.save_indicators(indicators_df, symbol, timeframe)

        print(f"✓ Update complete: {len(df_with_indicators)} candles with {len(indicator_cols)} indicators")

        return df_with_indicators

    def update_all_configured(self, force_refresh: bool = False):
        """
        Update data for all configured symbols and timeframes

        Args:
            force_refresh: If True, fetch all data from scratch
        """
        print("\n" + "="*60)
        print("STARTING FULL DATA PIPELINE UPDATE")
        print("="*60)

        results = {}

        # Update crypto data
        if DATA_SOURCES["crypto"]["enabled"]:
            results["crypto"] = {}
            for symbol in DATA_SOURCES["crypto"]["symbols"]:
                results["crypto"][symbol] = {}
                for timeframe in DATA_SOURCES["crypto"]["timeframes"]:
                    df = self.update_data(symbol, timeframe, "crypto", force_refresh)
                    results["crypto"][symbol][timeframe] = df

        # Update stock data
        if DATA_SOURCES["stocks"]["enabled"]:
            results["stocks"] = {}
            for symbol in DATA_SOURCES["stocks"]["symbols"]:
                results["stocks"][symbol] = {}
                for timeframe in DATA_SOURCES["stocks"]["timeframes"]:
                    df = self.update_data(symbol, timeframe, "stocks", force_refresh)
                    results["stocks"][symbol][timeframe] = df

        print("\n" + "="*60)
        print("PIPELINE UPDATE COMPLETE")
        print("="*60)

        return results

    def get_data_with_indicators(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Retrieve data with indicators from database

        Args:
            symbol: Trading pair or ticker
            timeframe: Candle timeframe

        Returns:
            DataFrame with OHLCV data and indicators
        """
        # Load OHLCV data
        df = self.db.load_ohlcv(symbol, timeframe)

        if df.empty:
            return df

        # Load indicators
        indicators_df = self.db.load_indicators(symbol, timeframe)

        if not indicators_df.empty:
            # Merge on timestamp
            df = df.join(indicators_df, how='left')

        return df

    def get_data_summary(self) -> pd.DataFrame:
        """
        Get summary of all data in database

        Returns:
            DataFrame with summary information
        """
        query = """
            SELECT
                symbol,
                timeframe,
                source,
                COUNT(*) as candle_count,
                datetime(MIN(timestamp), 'unixepoch') as start_date,
                datetime(MAX(timestamp), 'unixepoch') as end_date
            FROM ohlcv_data
            GROUP BY symbol, timeframe, source
            ORDER BY symbol, timeframe
        """

        return pd.read_sql_query(query, self.db.conn)

    def close(self):
        """Close database connection"""
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    """Main entry point for data pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description="Trading Bot Data Pipeline")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh all data from scratch"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show data summary only"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="Update specific symbol only"
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        help="Update specific timeframe only"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["crypto", "stocks"],
        help="Update specific source only"
    )

    args = parser.parse_args()

    pipeline = DataPipeline()

    try:
        if args.summary:
            # Show summary
            print("\nData Summary:")
            print("="*80)
            summary = pipeline.get_data_summary()
            if summary.empty:
                print("No data in database")
            else:
                print(summary.to_string(index=False))

        elif args.symbol and args.timeframe and args.source:
            # Update specific symbol
            pipeline.update_data(args.symbol, args.timeframe, args.source, args.refresh)

        else:
            # Update all configured data
            pipeline.update_all_configured(args.refresh)

            # Show summary
            print("\n" + "="*60)
            print("DATA SUMMARY")
            print("="*60)
            summary = pipeline.get_data_summary()
            print(summary.to_string(index=False))

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
