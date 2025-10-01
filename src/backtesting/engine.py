"""
Backtesting Engine for ML Trading Models
Vectorized implementation with transaction costs
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from pathlib import Path


class BacktestEngine:
    """Vectorized backtesting engine with realistic transaction costs"""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005,  # 0.05% slippage
    ):
        """
        Initialize backtesting engine

        Args:
            initial_capital: Starting capital in dollars
            transaction_cost: Transaction cost as fraction (0.001 = 0.1%)
            slippage: Slippage as fraction (0.0005 = 0.05%)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage

    def run_backtest(
        self,
        prices: pd.Series,
        signals: pd.Series,
        signal_probabilities: Optional[pd.Series] = None,
        confidence_threshold: Optional[float] = None
    ) -> Dict:
        """
        Run backtest on price data with given signals

        Args:
            prices: Series of prices (close prices)
            signals: Series of signals (1=long, 0=cash, -1=short)
            signal_probabilities: Optional confidence scores for signals
            confidence_threshold: Optional threshold for filtering signals

        Returns:
            Dictionary with backtest results
        """
        # Align data
        common_idx = prices.index.intersection(signals.index)
        prices = prices.loc[common_idx]
        signals = signals.loc[common_idx]

        # Apply confidence filter if provided
        if signal_probabilities is not None and confidence_threshold is not None:
            signal_probabilities = signal_probabilities.loc[common_idx]
            low_confidence_mask = signal_probabilities < confidence_threshold
            signals = signals.copy()
            signals[low_confidence_mask] = 0  # Stay in cash when low confidence

        # Calculate returns
        returns = prices.pct_change().fillna(0)

        # Calculate positions (shift signals to avoid lookahead bias)
        positions = signals.shift(1).fillna(0)

        # Calculate trades (position changes)
        trades = positions.diff().fillna(0)

        # Calculate strategy returns
        strategy_returns = positions * returns

        # Calculate transaction costs
        trade_costs = np.abs(trades) * (self.transaction_cost + self.slippage)

        # Net returns after costs
        net_returns = strategy_returns - trade_costs

        # Calculate equity curve
        equity_curve = (1 + net_returns).cumprod() * self.initial_capital

        # Calculate metrics
        metrics = self._calculate_metrics(
            net_returns, equity_curve, trades, positions, prices
        )

        # Store results
        results = {
            'equity_curve': equity_curve,
            'returns': net_returns,
            'positions': positions,
            'trades': trades,
            'signals': signals,
            'metrics': metrics,
            'prices': prices
        }

        return results

    def _calculate_metrics(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        trades: pd.Series,
        positions: pd.Series,
        prices: pd.Series
    ) -> Dict:
        """Calculate performance metrics"""

        # Total return
        total_return = (equity_curve.iloc[-1] / self.initial_capital - 1) * 100

        # Buy and hold return
        buy_hold_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100

        # Annualized return (assuming hourly data)
        trading_hours_per_year = 252 * 6.5  # Trading days * hours per day
        periods = len(returns)
        years = periods / trading_hours_per_year
        annualized_return = ((equity_curve.iloc[-1] / self.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

        # Sharpe ratio (annualized)
        risk_free_rate = 0.02  # 2% annual risk-free rate
        excess_returns = returns - (risk_free_rate / trading_hours_per_year)
        sharpe_ratio = np.sqrt(trading_hours_per_year) * (excess_returns.mean() / excess_returns.std()) if excess_returns.std() > 0 else 0

        # Maximum drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # Number of trades
        num_trades = (np.abs(trades) > 0).sum()

        # Win rate
        trade_returns = returns[np.abs(trades) > 0]
        if len(trade_returns) > 0:
            wins = (trade_returns > 0).sum()
            win_rate = (wins / len(trade_returns)) * 100
        else:
            win_rate = 0

        # Profit factor
        gross_profit = trade_returns[trade_returns > 0].sum()
        gross_loss = abs(trade_returns[trade_returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        # Average trade
        avg_trade_return = trade_returns.mean() * 100 if len(trade_returns) > 0 else 0

        # Time in market
        time_in_market = (positions != 0).sum() / len(positions) * 100

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'buy_hold_return': buy_hold_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_return': avg_trade_return,
            'time_in_market': time_in_market,
            'final_equity': equity_curve.iloc[-1]
        }

    def compare_strategies(
        self,
        prices: pd.Series,
        strategies: Dict[str, pd.Series],
        probabilities: Optional[Dict[str, pd.Series]] = None,
        confidence_threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Compare multiple strategies

        Args:
            prices: Series of prices
            strategies: Dict of {strategy_name: signals}
            probabilities: Optional dict of {strategy_name: probabilities}
            confidence_threshold: Optional confidence threshold

        Returns:
            DataFrame comparing all strategies
        """
        results = {}

        for name, signals in strategies.items():
            proba = probabilities.get(name) if probabilities else None
            backtest = self.run_backtest(
                prices, signals, proba, confidence_threshold
            )
            results[name] = backtest['metrics']

        # Convert to DataFrame
        comparison_df = pd.DataFrame(results).T

        # Sort by Sharpe ratio
        comparison_df = comparison_df.sort_values('sharpe_ratio', ascending=False)

        return comparison_df


class SimpleStrategies:
    """Simple baseline strategies for comparison"""

    @staticmethod
    def buy_and_hold(prices: pd.Series) -> pd.Series:
        """Buy and hold strategy (always long)"""
        return pd.Series(1, index=prices.index)

    @staticmethod
    def rsi_strategy(df: pd.DataFrame, rsi_col: str = 'RSI_14', oversold: float = 30, overbought: float = 70) -> pd.Series:
        """
        Simple RSI crossover strategy

        Args:
            df: DataFrame with RSI column
            rsi_col: Name of RSI column
            oversold: RSI oversold threshold (buy signal)
            overbought: RSI overbought threshold (sell signal)

        Returns:
            Series of signals (1=long, 0=cash)
        """
        if rsi_col not in df.columns:
            raise ValueError(f"RSI column '{rsi_col}' not found in DataFrame")

        signals = pd.Series(0, index=df.index)
        rsi = df[rsi_col]

        # Buy when RSI crosses above oversold
        # Sell when RSI crosses above overbought
        position = 0
        for i in range(1, len(df)):
            if rsi.iloc[i] > oversold and rsi.iloc[i-1] <= oversold:
                position = 1  # Buy signal
            elif rsi.iloc[i] > overbought:
                position = 0  # Sell signal
            signals.iloc[i] = position

        return signals

    @staticmethod
    def moving_average_crossover(df: pd.DataFrame, fast_period: int = 20, slow_period: int = 50) -> pd.Series:
        """
        Moving average crossover strategy

        Args:
            df: DataFrame with close prices
            fast_period: Fast MA period
            slow_period: Slow MA period

        Returns:
            Series of signals (1=long, 0=cash)
        """
        if 'close' not in df.columns:
            raise ValueError("DataFrame must have 'close' column")

        fast_ma = df['close'].rolling(window=fast_period).mean()
        slow_ma = df['close'].rolling(window=slow_period).mean()

        # Long when fast MA > slow MA
        signals = (fast_ma > slow_ma).astype(int)

        return signals
