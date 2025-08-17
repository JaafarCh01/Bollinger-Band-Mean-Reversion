#!/usr/bin/env python3
"""
Simple script to run the Bollinger Band Mean Reversion Strategy
"""

from bollinger_strategy import BollingerMeanReversionStrategy

def main():
    print("🚀 Bollinger Band Mean Reversion Strategy")
    print("=" * 50)
    
    # Initialize strategy with parameters from Guide.md
    strategy = BollingerMeanReversionStrategy(
        bb_period=20,      # Bollinger Band period
        bb_std_dev=2,      # Standard deviations for bands
        rsi_period=14,     # RSI period
        rsi_oversold=30,   # RSI oversold threshold
        rsi_overbought=70  # RSI overbought threshold
    )
    
    # Run complete analysis
    data_file = "BTCUSD_15 Mins_Ask_2025.01.01_2025.08.16.csv"
    
    try:
        data, results = strategy.run_complete_analysis(data_file, initial_capital=10000)
        
        print("\n📊 Strategy Analysis Complete!")
        print(f"📈 Data points analyzed: {len(data)}")
        print(f"💰 Final equity: ${results['Final Equity']:,.2f}")
        print(f"📉 Max drawdown: {results['Max Drawdown (%)']:.2f}%")
        print(f"🎯 Win rate: {results['Win Rate (%)']:.1f}%")
        
        # Show a sample of recent data
        print("\n📋 Recent Data Sample:")
        print(data[['Close', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'RSI', 'Position']].tail())
        
        # Plot results (this will show the chart)
        print("\n📊 Generating charts...")
        strategy.plot_results()
        
    except Exception as e:
        print(f"❌ Error running strategy: {e}")
        print("Make sure the CSV file is in the same directory")

if __name__ == "__main__":
    main()
