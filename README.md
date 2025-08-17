# 🎯 Bollinger Band Mean Reversion Strategy

A comprehensive implementation of a Bollinger Band mean reversion trading strategy for cryptocurrency markets, specifically designed for BTC/USD 15-minute data.

## 📊 Strategy Overview

- **Strategy Type**: Mean Reversion using Bollinger Bands + RSI confirmation
- **Target Win Rate**: 55-70%
- **Risk Profile**: Moderate (target max drawdown <15%)
- **Best Market Conditions**: Range-bound/sideways markets
- **Timeframe**: 15-minute candles

## 🛠️ Features

✅ **Complete Implementation**
- Bollinger Bands calculation (20-period SMA, 2 std dev)
- RSI indicator (14-period with 30/70 thresholds)
- Signal generation with entry/exit logic
- Comprehensive backtesting framework
- Performance metrics calculation
- Advanced visualization plots

✅ **Key Metrics Tracked**
- Total return vs market return
- Maximum drawdown
- Win rate and total trades
- Sharpe ratio
- Equity curve analysis

✅ **Professional Visualizations**
- Price chart with Bollinger Bands
- RSI oscillator with overbought/oversold levels
- Position tracking
- Equity curve comparison (strategy vs buy-and-hold)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Strategy
```bash
python run_strategy.py
```

### 3. Advanced Usage
```python
from bollinger_strategy import BollingerMeanReversionStrategy

# Initialize with custom parameters
strategy = BollingerMeanReversionStrategy(
    bb_period=20,      # Bollinger Band period
    bb_std_dev=2,      # Standard deviations
    rsi_period=14,     # RSI period
    rsi_oversold=30,   # RSI oversold threshold
    rsi_overbought=70  # RSI overbought threshold
)

# Run complete analysis
data, results = strategy.run_complete_analysis(
    "BTCUSD_15 Mins_Ask_2025.01.01_2025.08.16.csv", 
    initial_capital=10000
)

# Generate plots
strategy.plot_results()
```

## 📈 Trading Logic

### Entry Signals
**Long Position (Buy)**:
- Price touches or breaks below the lower Bollinger Band
- RSI is oversold (≤ 30)

**Short Position (Sell)**:
- Price touches or breaks above the upper Bollinger Band  
- RSI is overbought (≥ 70)

### Exit Signals
- **Long Exit**: Price crosses back above the middle Bollinger Band (20-period SMA)
- **Short Exit**: Price crosses back below the middle Bollinger Band (20-period SMA)

## 📊 Performance Metrics

The strategy calculates and displays:

- **Total Return (%)**: Overall strategy performance
- **Market Return (%)**: Buy-and-hold benchmark performance  
- **Excess Return (%)**: Strategy outperformance vs market
- **Max Drawdown (%)**: Largest peak-to-trough decline
- **Win Rate (%)**: Percentage of profitable trades
- **Total Trades**: Number of completed round-trip trades
- **Sharpe Ratio**: Risk-adjusted return measure
- **Final Equity**: Ending portfolio value

## 📁 File Structure

```
Bollinger Band Mean Reversion/
├── Guide.md                           # Strategy documentation
├── BTCUSD_15 Mins_Ask_2025.01.01_2025.08.16.csv  # Historical data
├── bollinger_strategy.py              # Main strategy implementation
├── run_strategy.py                    # Simple execution script
├── requirements.txt                   # Python dependencies
└── README.md                         # This file
```

## ⚙️ Customization

You can easily customize the strategy parameters:

```python
strategy = BollingerMeanReversionStrategy(
    bb_period=20,        # Bollinger Band lookback period
    bb_std_dev=2.0,      # Standard deviation multiplier
    rsi_period=14,       # RSI calculation period
    rsi_oversold=30,     # RSI oversold threshold
    rsi_overbought=70    # RSI overbought threshold
)
```

## 📊 Visualization Features

The strategy generates comprehensive charts showing:

1. **Price & Bollinger Bands**: BTC price with upper/lower bands and trading signals
2. **RSI Oscillator**: RSI values with overbought/oversold zones highlighted
3. **Position Tracking**: Visual representation of long/short/neutral positions
4. **Equity Curve**: Strategy performance vs buy-and-hold comparison

## ⚠️ Risk Disclaimer

This strategy is for educational and research purposes only. Past performance does not guarantee future results. Always conduct thorough testing and risk management before using any trading strategy with real capital.

## 🔧 Technical Requirements

- Python 3.7+
- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0

## 📞 Support

For questions or improvements, refer to the detailed implementation in `bollinger_strategy.py` or the strategy documentation in `Guide.md`.

---

**Happy Trading! 🚀📈**
