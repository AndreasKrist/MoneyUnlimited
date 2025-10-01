"""
Professional Trading Bot Dashboard with Streamlit
Real-time predictions, backtesting results, and paper trading
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import joblib
from datetime import datetime, timedelta

from src.data.database import TradingDatabase
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from train_improved import ImprovedFeatureEngineering
from src.backtesting.engine import BacktestEngine

# Page config
st.set_page_config(
    page_title="ML Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #334155;
    }
    .stMetric label {
        color: #94a3b8 !important;
        font-weight: 600;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-size: 2rem !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #94a3b8 !important;
    }
    h1 {
        color: #3b82f6;
    }
    h2, h3 {
        color: #e2e8f0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 25px;
        border-radius: 15px;
        border: 2px solid #475569;
        margin: 10px 0;
    }
    .signal-buy {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .signal-sell {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    [data-testid="stSidebar"] {
        background-color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def load_data(symbol="SPY", timeframe="1h"):
    """Load OHLCV and indicators from database"""
    db = TradingDatabase()
    ohlcv = db.load_ohlcv(symbol, timeframe)
    indicators = db.load_indicators(symbol, timeframe)

    if ohlcv is None or len(ohlcv) == 0:
        return None

    if indicators is not None and len(indicators) > 0:
        df = ohlcv.join(indicators, how='inner')
    else:
        df = ohlcv

    return df.sort_index()


@st.cache_resource
def load_models():
    """Load trained ML models"""
    models_dir = Path("models/improved")

    rf_model = RandomForestModel()
    rf_model.load(models_dir / "random_forest_improved.joblib")

    xgb_model = XGBoostModel()
    xgb_model.load(models_dir / "xgboost_improved.joblib")

    return rf_model, xgb_model


def generate_predictions(df, rf_model, xgb_model):
    """Generate model predictions"""
    # Create features
    feature_eng = ImprovedFeatureEngineering(prediction_horizon=2)
    features, target = feature_eng.create_all_features(df)
    features = features.fillna(0)

    # Use model's feature names
    model_features = rf_model.feature_names

    # Add missing features
    for feat in model_features:
        if feat not in features.columns:
            features[feat] = 0

    # Select features
    X = features[model_features]

    # Generate predictions
    rf_proba = rf_model.predict_proba(X)
    xgb_proba = xgb_model.predict_proba(X)
    ensemble_proba = (rf_proba + xgb_proba) / 2

    # Create results dataframe
    predictions = pd.DataFrame({
        'rf_proba_up': rf_proba[:, 1],
        'rf_proba_down': rf_proba[:, 0],
        'rf_signal': (rf_proba[:, 1] > 0.5).astype(int),
        'xgb_proba_up': xgb_proba[:, 1],
        'xgb_proba_down': xgb_proba[:, 0],
        'xgb_signal': (xgb_proba[:, 1] > 0.5).astype(int),
        'ensemble_proba_up': ensemble_proba[:, 1],
        'ensemble_proba_down': ensemble_proba[:, 0],
        'ensemble_signal': (ensemble_proba[:, 1] > 0.5).astype(int),
        'ensemble_confidence': np.max(ensemble_proba, axis=1),
    }, index=X.index)

    return predictions


def create_price_chart(df, predictions, lookback=500):
    """Create interactive price chart with predictions"""
    # Last N candles
    df_plot = df.iloc[-lookback:].copy()
    pred_plot = predictions.loc[df_plot.index]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price with Predictions', 'Model Confidence'),
        row_heights=[0.7, 0.3]
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df_plot.index,
            open=df_plot['open'],
            high=df_plot['high'],
            low=df_plot['low'],
            close=df_plot['close'],
            name='Price'
        ),
        row=1, col=1
    )

    # Add prediction markers
    buy_signals = pred_plot[pred_plot['ensemble_signal'] == 1]
    sell_signals = pred_plot[pred_plot['ensemble_signal'] == 0]

    fig.add_trace(
        go.Scatter(
            x=buy_signals.index,
            y=df_plot.loc[buy_signals.index, 'close'],
            mode='markers',
            marker=dict(symbol='triangle-up', size=10, color='green'),
            name='Buy Signal'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=sell_signals.index,
            y=df_plot.loc[sell_signals.index, 'close'],
            mode='markers',
            marker=dict(symbol='triangle-down', size=10, color='red'),
            name='Sell Signal'
        ),
        row=1, col=1
    )

    # Confidence chart
    fig.add_trace(
        go.Scatter(
            x=pred_plot.index,
            y=pred_plot['ensemble_confidence'],
            fill='tozeroy',
            name='Confidence',
            line=dict(color='purple')
        ),
        row=2, col=1
    )

    # Add confidence threshold line
    fig.add_hline(y=0.6, line_dash="dash", line_color="orange", row=2, col=1)

    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(
        height=700,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )

    return fig


def create_confidence_gauge(confidence, label):
    """Create gauge chart for confidence"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': label},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 60
            }
        }
    ))

    fig.update_layout(height=250)
    return fig


def create_equity_curve_chart(backtest_results):
    """Create equity curve comparison"""
    fig = go.Figure()

    for name, result in backtest_results.items():
        equity = result['equity_curve']
        fig.add_trace(go.Scatter(
            x=equity.index,
            y=equity.values,
            name=name,
            mode='lines',
            line=dict(width=2)
        ))

    fig.update_layout(
        title="Equity Curves ($10,000 Initial Capital)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )

    return fig


def create_feature_importance_chart(model, model_name, top_n=15):
    """Create feature importance bar chart"""
    importance_df = model.get_feature_importance(top_n=top_n)

    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title=f'{model_name} - Top {top_n} Features',
        labels={'importance': 'Importance', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='Blues'
    )

    fig.update_layout(
        height=500,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'},
        template='plotly_white'
    )

    return fig


def main():
    # Title
    st.title("🤖 ML Trading Bot Dashboard")
    st.markdown("---")

    # Sidebar
    st.sidebar.title("⚙️ Settings")

    # Asset selection
    asset_type = st.sidebar.radio("Asset Type", ["📊 Stocks", "₿ Crypto"])

    if asset_type == "📊 Stocks":
        symbol = st.sidebar.selectbox("Symbol", ["SPY", "QQQ", "AAPL", "MSFT", "TSLA"])
    else:
        symbol = st.sidebar.selectbox("Symbol", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"])

    timeframe = st.sidebar.selectbox("Timeframe", ["1h", "4h", "1d"], index=0)

    mode = st.sidebar.radio("Mode", ["📊 Analysis", "📈 Live Predictions", "🎯 Paper Trading"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Model Info")
    st.sidebar.info("**Ensemble Model**\n\n• Random Forest (300 trees)\n• XGBoost (200 trees)\n• 25 features selected\n• 2-hour prediction horizon")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Quick Stats")

    # Show available data count
    try:
        db = TradingDatabase()
        all_symbols = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA", "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
        available_count = 0
        for sym in all_symbols:
            test_df = db.load_ohlcv(sym, timeframe)
            if test_df is not None and len(test_df) > 0:
                available_count += 1
        st.sidebar.metric("Available Assets", f"{available_count}/{len(all_symbols)}")
    except:
        pass

    # Load data
    with st.spinner("Loading data..."):
        df = load_data(symbol, timeframe)

        if df is None or len(df) == 0:
            st.error(f"No data found for {symbol} {timeframe}")
            st.stop()

        rf_model, xgb_model = load_models()
        predictions = generate_predictions(df, rf_model, xgb_model)

    st.success(f"✅ Loaded {len(df)} candles")

    # Main content based on mode
    if mode == "📊 Analysis":
        show_analysis_mode(df, predictions, rf_model, xgb_model)
    elif mode == "📈 Live Predictions":
        show_live_predictions_mode(df, predictions)
    else:
        show_paper_trading_mode(df, predictions)


def show_analysis_mode(df, predictions, rf_model, xgb_model):
    """Analysis mode - backtest results and model diagnostics"""

    # Metrics row
    st.header("📊 Performance Metrics")

    # Load backtest results
    backtest_file = Path("backtests/backtest_results.csv")
    if backtest_file.exists():
        backtest_df = pd.read_csv(backtest_file, index_col=0)

        col1, col2, col3, col4, col5 = st.columns(5)

        # Buy & Hold
        bh_metrics = backtest_df.loc['Buy & Hold']
        with col1:
            st.metric(
                "Buy & Hold Return",
                f"{bh_metrics['total_return']:.2f}%",
                f"Sharpe: {bh_metrics['sharpe_ratio']:.2f}"
            )

        # Ensemble
        ens_metrics = backtest_df.loc['Ensemble']
        with col2:
            st.metric(
                "Ensemble Return",
                f"{ens_metrics['total_return']:.2f}%",
                f"Sharpe: {ens_metrics['sharpe_ratio']:.2f}",
                delta_color="inverse" if ens_metrics['total_return'] < 0 else "normal"
            )

        # XGBoost
        xgb_metrics = backtest_df.loc['XGBoost']
        with col3:
            st.metric(
                "XGBoost Return",
                f"{xgb_metrics['total_return']:.2f}%",
                f"Sharpe: {xgb_metrics['sharpe_ratio']:.2f}",
                delta_color="inverse" if xgb_metrics['total_return'] < 0 else "normal"
            )

        with col4:
            st.metric(
                "Win Rate",
                f"{ens_metrics['win_rate']:.1f}%",
                f"{int(ens_metrics['num_trades'])} trades"
            )

        with col5:
            st.metric(
                "Max Drawdown",
                f"{ens_metrics['max_drawdown']:.2f}%",
                f"PF: {ens_metrics['profit_factor']:.2f}"
            )

        st.markdown("---")

        # Full comparison table
        st.subheader("📋 Strategy Comparison Table")
        st.dataframe(
            backtest_df.style.format({
                'total_return': '{:.2f}%',
                'annualized_return': '{:.2f}%',
                'sharpe_ratio': '{:.2f}',
                'max_drawdown': '{:.2f}%',
                'win_rate': '{:.2f}%',
                'num_trades': '{:.0f}',
                'profit_factor': '{:.2f}'
            }).background_gradient(subset=['sharpe_ratio'], cmap='RdYlGn'),
            use_container_width=True
        )
    else:
        st.warning("No backtest results found. Run `python run_backtest.py` first.")

    st.markdown("---")

    # Feature importance
    st.header("🎯 Feature Importance")
    col1, col2 = st.columns(2)

    with col1:
        fig_rf = create_feature_importance_chart(rf_model, "Random Forest")
        st.plotly_chart(fig_rf, use_container_width=True)

    with col2:
        fig_xgb = create_feature_importance_chart(xgb_model, "XGBoost")
        st.plotly_chart(fig_xgb, use_container_width=True)


def show_live_predictions_mode(df, predictions):
    """Live predictions mode - real-time signals"""

    # Current prediction
    latest = predictions.iloc[-1]
    latest_price = df['close'].iloc[-1]
    latest_time = df.index[-1]

    st.header("🔮 Latest Prediction")

    # Create styled prediction cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        price_change = df['close'].pct_change().iloc[-1] * 100
        st.metric(
            "💰 Current Price",
            f"${latest_price:.2f}",
            f"{price_change:.2f}%",
            delta_color="normal"
        )
        st.caption(f"📅 {latest_time.strftime('%Y-%m-%d %H:%M')}")

    with col2:
        signal_text = "BUY" if latest['ensemble_signal'] == 1 else "SELL"
        signal_emoji = "🟢" if latest['ensemble_signal'] == 1 else "🔴"
        signal_class = "signal-buy" if latest['ensemble_signal'] == 1 else "signal-sell"

        st.markdown(f"""
        <div class="{signal_class}">
            {signal_emoji} {signal_text}
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"🎯 Confidence: {latest['ensemble_confidence']*100:.1f}%")

    with col3:
        high_conf = latest['ensemble_confidence'] > 0.6
        quality_emoji = "⭐" if high_conf else "⚠️"
        quality_text = "HIGH CONFIDENCE" if high_conf else "LOW CONFIDENCE"
        quality_color = "#10b981" if high_conf else "#f59e0b"

        st.markdown(f"""
        <div class="prediction-card" style="border-color: {quality_color};">
            <h3 style="margin: 0; color: {quality_color};">{quality_emoji} {quality_text}</h3>
            <p style="margin: 5px 0 0 0; color: #94a3b8;">{'✅ Trade Recommended' if high_conf else '❌ Skip This Trade'}</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        # Show model agreement
        rf_signal = latest['rf_signal']
        xgb_signal = latest['xgb_signal']
        agreement = (rf_signal == xgb_signal)
        agreement_emoji = "✅" if agreement else "⚠️"
        agreement_text = "Models Agree" if agreement else "Models Disagree"
        agreement_color = "#10b981" if agreement else "#f59e0b"

        st.markdown(f"""
        <div class="prediction-card" style="border-color: {agreement_color};">
            <h3 style="margin: 0; color: {agreement_color};">{agreement_emoji} {agreement_text}</h3>
            <p style="margin: 5px 0 0 0; color: #94a3b8;">RF: {rf_signal} | XGB: {xgb_signal}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Gauges
    st.subheader("📊 Model Confidence Levels")
    col1, col2, col3 = st.columns(3)

    with col1:
        rf_conf = max(latest['rf_proba_up'], latest['rf_proba_down'])
        st.plotly_chart(create_confidence_gauge(rf_conf, "Random Forest"), use_container_width=True)

    with col2:
        xgb_conf = max(latest['xgb_proba_up'], latest['xgb_proba_down'])
        st.plotly_chart(create_confidence_gauge(xgb_conf, "XGBoost"), use_container_width=True)

    with col3:
        st.plotly_chart(create_confidence_gauge(latest['ensemble_confidence'], "Ensemble"), use_container_width=True)

    st.markdown("---")

    # Price chart with predictions
    st.subheader("📈 Price Chart with Signals")
    fig = create_price_chart(df, predictions, lookback=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Recent predictions table
    st.subheader("📋 Recent Predictions")
    recent = predictions.tail(20).copy()
    recent['price'] = df.loc[recent.index, 'close']
    recent['signal'] = recent['ensemble_signal'].map({1: '🟢 BUY', 0: '🔴 SELL'})
    recent['confidence'] = (recent['ensemble_confidence'] * 100).round(1).astype(str) + '%'

    display_cols = ['price', 'signal', 'confidence', 'rf_proba_up', 'xgb_proba_up', 'ensemble_proba_up']
    st.dataframe(
        recent[display_cols].sort_index(ascending=False).style.format({
            'price': '${:.2f}',
            'rf_proba_up': '{:.1%}',
            'xgb_proba_up': '{:.1%}',
            'ensemble_proba_up': '{:.1%}'
        }),
        use_container_width=True
    )


def show_paper_trading_mode(df, predictions):
    """Paper trading mode - simulate live trading"""

    st.header("🎯 Paper Trading Simulator")

    st.info("Paper trading simulates real-time trading with virtual money. No real funds at risk!")

    # Settings
    col1, col2, col3 = st.columns(3)
    with col1:
        initial_capital = st.number_input("Initial Capital ($)", value=10000, step=1000)
    with col2:
        confidence_threshold = st.slider("Min Confidence", 0.5, 0.9, 0.6, 0.05)
    with col3:
        holding_period = st.number_input("Min Holding (hours)", value=4, step=1)

    st.markdown("---")

    # Simulate paper trading
    if st.button("🚀 Run Paper Trading Simulation", type="primary"):
        with st.spinner("Simulating trades..."):
            # Use ensemble signals with confidence filter
            signals = predictions['ensemble_signal'].copy()
            signals[predictions['ensemble_confidence'] < confidence_threshold] = 0

            # Run backtest
            backtester = BacktestEngine(
                initial_capital=initial_capital,
                transaction_cost=0.001,
                slippage=0.0005
            )

            result = backtester.run_backtest(
                df['close'],
                signals,
                predictions['ensemble_confidence'],
                confidence_threshold
            )

            metrics = result['metrics']

            # Display results
            st.success("✅ Simulation Complete!")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{metrics['total_return']:.2f}%")
            with col2:
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            with col3:
                st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
            with col4:
                st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")

            # Equity curve
            st.subheader("💰 Equity Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=result['equity_curve'].index,
                y=result['equity_curve'].values,
                fill='tozeroy',
                name='Portfolio Value',
                line=dict(color='green', width=2)
            ))
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=400,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Trade log
            st.subheader("📝 Trade Log")
            trades_df = pd.DataFrame({
                'timestamp': result['trades'].index[result['trades'] != 0],
                'action': result['trades'][result['trades'] != 0].map({1: 'BUY', -1: 'SELL', 2: 'BUY'}),
                'price': df.loc[result['trades'].index[result['trades'] != 0], 'close']
            })

            if len(trades_df) > 0:
                st.dataframe(
                    trades_df.tail(50).style.format({'price': '${:.2f}'}),
                    use_container_width=True
                )
            else:
                st.warning("No trades executed. Try lowering the confidence threshold.")


if __name__ == "__main__":
    main()
