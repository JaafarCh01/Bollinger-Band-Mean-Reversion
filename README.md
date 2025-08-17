# 🚀 Market-Adaptive Hybrid Trading Strategy

A sophisticated cryptocurrency trading system that automatically switches between **Bollinger Band Mean Reversion** (range markets) and **EMA Crossover** (trending markets) based on real-time **ADX market regime detection**.

## 🎯 Strategy Overview

- **Strategy Type**: Market-Adaptive Hybrid (Bollinger Bands + EMA Crossover)
- **Market Regime Detection**: ADX-based automatic switching
- **Target Win Rate**: 60-75% (combined across regimes)  
- **Risk Profile**: Moderate (max drawdown target <20%)
- **Timeframes**: Hourly BTC/USD data
- **Assets**: Cryptocurrencies (BTC, ETH, etc.)

## 🧠 How It Works

### 🔄 Automatic Strategy Switching
The system uses **ADX (Average Directional Index)** to detect market conditions:

- **📈 Trending Markets** (ADX > 25): Uses **EMA Crossover Strategy**
- **📊 Range-Bound Markets** (ADX < 20): Uses **Bollinger Band Strategy**  
- **⏸️ Transition Period** (ADX 20-25): No trading (risk management)

### 📊 Component Strategies

#### 1. Bollinger Band Mean Reversion (Range Markets)
- **Indicators**: Bollinger Bands (20-period, 2σ) + RSI (14-period)
- **Entry**: Price below lower band + RSI <30 + bullish candle + volume spike
- **Exit**: Price reaches middle band or RSI >70

#### 2. EMA Crossover (Trending Markets)  
- **Indicators**: EMA 5, 21, 55 periods
- **Entry**: Price > EMA55 + EMA21 > EMA55 + EMA5 < EMA21 + bullish candle
- **Exit**: EMA5 crosses above EMA21

## 🛠️ Complete Feature Set

✅ **Market-Adaptive Core**
- ADX-based regime detection and automatic strategy switching
- Real-time position closure on regime changes
- Comprehensive regime-specific performance tracking

✅ **Advanced Risk Management**
- Dynamic position sizing based on drawdown
- Stop-loss protection (2% default)
- Daily trade limits (3 trades max)
- Daily loss limits (5% max)

✅ **Professional Analytics**
- Regime-specific win rates and returns
- Market regime distribution analysis
- Strategy switching frequency tracking
- Comprehensive performance metrics

✅ **Advanced Visualizations**
- Price charts with regime background coloring
- ADX trend strength indicator
- Active strategy timeline
- Position tracking and equity curves
- Regime-specific performance breakdowns

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Hybrid Strategy
```bash
python run_strategy.py
```

### 3. Run Individual Strategy Comparison
```bash
python run_strategy.py --individual
```

### 4. Advanced Usage
```python
from market_adaptive_strategy import MarketAdaptiveStrategy

# Initialize with custom parameters
strategy = MarketAdaptiveStrategy(
    # ADX Parameters
    adx_period=14,
    trend_threshold=25,    # ADX > 25 = trending
    range_threshold=20,    # ADX < 20 = range-bound
    
    # Bollinger Parameters (range markets)
    bb_period=20,
    bb_std_dev=2,
    rsi_oversold=30,
    rsi_overbought=70,
    
    # EMA Parameters (trending markets)
    ema_short=5,
    ema_medium=21,
    ema_long=55,
    
    # Risk Management
    risk_per_trade=0.01,      # 1% risk per trade
    stop_loss_pct=0.02,       # 2% stop loss
    max_trades_per_day=3,     # Max 3 trades/day
    daily_loss_limit=0.05     # 5% daily loss limit
)

# Run complete analysis
data, results = strategy.run_complete_analysis(
    "BTCUSD_Hourly_Ask_2024.01.01_2024.12.31.csv", 
    initial_capital=10000
)

# Generate comprehensive charts
strategy.plot_comprehensive_results()
```

## 📈 Trading Logic

### 🔍 Market Regime Detection
```python
if ADX > 25:
    regime = "trending"      # Use EMA Crossover
elif ADX < 20:
    regime = "range"         # Use Bollinger Bands  
else:
    regime = "transition"    # No trading
```

### 📊 Strategy-Specific Signals

#### Bollinger Band Signals (Range Markets)
- **Long Entry**: Price ≤ Lower Band + RSI ≤ 30 + Bullish Candle + Volume Spike
- **Short Entry**: Price ≥ Upper Band + RSI ≥ 70 + Bearish Candle + Volume Spike
- **Exit**: Price crosses middle band or time-based (10 bars)

#### EMA Crossover Signals (Trending Markets)
- **Long Entry**: Price > EMA55 + EMA21 > EMA55 + EMA5 < EMA21 + Bullish Candle
- **Exit**: EMA5 crosses above EMA21 or time-based (15 bars)

### ⚡ Regime Switching Logic
- **Immediate Position Closure**: When regime changes mid-trade
- **Strategy Switching**: Automatic transition between strategies
- **Risk Management**: Enhanced protection during transitions

## 📊 Performance Metrics

### 🎯 Overall Performance
- **Total Return (%)**: Strategy vs market performance
- **Excess Return (%)**: Strategy outperformance  
- **Max Drawdown (%)**: Largest peak-to-trough decline
- **Win Rate (%)**: Overall profitable trade percentage
- **Sharpe Ratio**: Risk-adjusted returns
- **Profit Factor**: Gross profit / gross loss

### 📈 Regime-Specific Analytics
- **Trending Market Performance**: EMA strategy results
- **Range Market Performance**: Bollinger strategy results  
- **Regime Distribution**: Time spent in each market condition
- **Strategy Switching Frequency**: Regime change analysis

## 📁 Project Structure

```
Market-Adaptive-Trading-System/
├── 📋 Guide.md                        # Comprehensive strategy guide
├── 📊 BTCUSD_Hourly_Ask_2024.01.01_2024.12.31.csv  # Hourly BTC data
├── 🧠 market_adaptive_strategy.py     # Main hybrid strategy
├── 📊 bollinger_strategy.py           # Bollinger Band component
├── 📈 ema_strategy.py                 # EMA Crossover component  
├── 🔍 regime_detector.py              # ADX market regime detection
├── 🚀 run_strategy.py                 # Strategy runner with options
├── 📦 requirements.txt                # Python dependencies
└── 📖 README.md                      # This file
```

## ⚙️ Parameter Optimization

### 🎛️ ADX Regime Detection
```python
adx_period = 14          # ADX calculation period
trend_threshold = 25     # Trending market threshold  
range_threshold = 20     # Range-bound market threshold
```

### 📊 Bollinger Band Tuning
```python
bb_period = 20          # Moving average period
bb_std_dev = 2          # Standard deviation multiplier
rsi_period = 14         # RSI calculation period
rsi_oversold = 30       # Oversold threshold
rsi_overbought = 70     # Overbought threshold
```

### 📈 EMA Crossover Tuning  
```python
ema_short = 5           # Fast EMA period
ema_medium = 21         # Medium EMA period
ema_long = 55           # Slow EMA period
```

### 🛡️ Risk Management
```python
risk_per_trade = 0.01        # Risk 1% per trade
stop_loss_pct = 0.02         # 2% stop loss
max_trades_per_day = 3       # Max 3 trades per day
daily_loss_limit = 0.05      # 5% daily loss limit
```

## 📊 Advanced Visualizations

The system generates 5 comprehensive chart panels:

1. **🎨 Price + Regime Background**: Color-coded market regimes with entry/exit signals
2. **📊 ADX Trend Strength**: Real-time regime detection with thresholds  
3. **⚡ Active Strategy Timeline**: Visual strategy switching indicator
4. **📍 Position Tracking**: Long/short/neutral position history
5. **💰 Equity Curves**: Strategy vs buy-and-hold performance comparison

## 🎯 Key Success Metrics

### 📊 Performance Targets
- **Overall Win Rate**: >60%
- **Profit Factor**: >1.8  
- **Max Drawdown**: <20%
- **Sharpe Ratio**: >1.2
- **Regime-Specific Win Rate**: >55% in each regime

### 🔧 Operational Targets
- **Regime Detection Accuracy**: >80%
- **Strategy Switching Latency**: <100ms
- **System Reliability**: >99.5% uptime

## 🔬 Research & Development

### 📈 Backtesting Features
- **Walk-forward analysis** capability
- **Monte Carlo simulation** ready
- **Parameter optimization** framework
- **Regime-specific backtesting**

### 🧪 Future Enhancements
- **Multi-asset support** (ETH, altcoins)
- **Machine learning** regime detection
- **Volatility-adjusted** position sizing
- **Options strategies** integration

## ⚠️ Risk Disclaimer

This strategy is for **educational and research purposes only**. Past performance does not guarantee future results. Cryptocurrency trading involves substantial risk of loss. Always:

- 🧪 **Backtest thoroughly** before live trading
- 💰 **Use proper position sizing** (never risk more than you can afford to lose)
- 📊 **Monitor performance** and adjust parameters as needed
- 🛡️ **Implement additional risk controls** for live trading

## 🔧 Technical Requirements

- **Python**: 3.7+
- **pandas**: ≥1.5.0  
- **numpy**: ≥1.21.0
- **matplotlib**: ≥3.5.0
- **scipy**: ≥1.9.0

## 📞 Support & Documentation

- 📋 **Strategy Guide**: See `Guide.md` for detailed implementation theory
- 🧠 **Core Logic**: `market_adaptive_strategy.py` contains main algorithm
- 📊 **Components**: Individual strategies in `bollinger_strategy.py` and `ema_strategy.py`
- 🔍 **Regime Detection**: ADX implementation in `regime_detector.py`

---

## 🎉 Ready to Trade Smarter?

**Experience the power of adaptive trading that automatically adjusts to market conditions!**

```bash
# Get started in 30 seconds
pip install -r requirements.txt
python run_strategy.py
```

**🚀 Happy Adaptive Trading! 📈✨**
