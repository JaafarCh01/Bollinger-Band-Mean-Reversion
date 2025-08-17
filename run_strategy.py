#!/usr/bin/env python3
"""
Market-Adaptive Hybrid Trading Strategy Runner
Combines Bollinger Band Mean Reversion (range markets) and EMA Crossover (trending markets)
with automatic ADX-based regime detection.
"""

from market_adaptive_strategy import MarketAdaptiveStrategy
from bollinger_strategy import BollingerMeanReversionStrategy
from ema_strategy import EMACrossoverStrategy
import sys

def run_hybrid_strategy():
    """Run the Market-Adaptive Hybrid Strategy"""
    print("🚀 Market-Adaptive Hybrid Trading Strategy")
    print("=" * 60)
    print("📊 Combining Bollinger Bands + EMA Crossover with ADX Regime Detection")
    print("=" * 60)
    
    # Initialize Market-Adaptive strategy with IMPROVED parameters
    strategy = MarketAdaptiveStrategy(
        # ADX Parameters for regime detection (UPDATED for crypto volatility)
        adx_period=14,
        trend_threshold=20,    # LOWERED: ADX > 20 = trending market (use EMA)
        range_threshold=15,    # LOWERED: ADX < 15 = range market (use Bollinger)
        
        # Bollinger Band Parameters (ENHANCED with 50 EMA filter)
        bb_period=20,
        bb_std_dev=2,
        rsi_period=14,
        rsi_oversold=40,       # IMPROVED: from 35 to 40 (better quality signals)
        rsi_overbought=65,     # RELAXED: from 70 to 65
        
        # EMA Parameters (for trending markets)
        ema_short=5,
        ema_medium=21,
        ema_long=55,
        
        # IMPROVED Risk Management
        risk_per_trade=0.02,      # INCREASED: Risk 2% per trade (was 1.5%)
        stop_loss_pct=0.03,       # INCREASED: 3% stop loss (was 2.5%)
        max_trades_per_day=5,     # INCREASED: Max 5 trades per day (was 3)
        daily_loss_limit=0.08,    # INCREASED: Stop if daily loss > 8% (was 5%)
        trailing_stop_pct=0.04,   # IMPROVED: 4% trailing stop (was 3%)
        
        # NEW: Regime stability parameters
        regime_stability_bars=3,  # Require 3 consecutive bars
        min_regime_duration=5     # Minimum 5 bars in regime
    )
    
    # Use hourly BTC data
    data_file = "BTCUSD_Hourly_Ask_2024.01.01_2024.12.31.csv"
    
    try:
        data, results = strategy.run_complete_analysis(data_file, initial_capital=10000)
        
        print("\n🎉 Market-Adaptive Strategy Analysis Complete!")
        print(f"📈 Data points analyzed: {len(data):,}")
        print(f"💰 Final equity: ${results['Final Equity']:,.2f}")
        print(f"📊 Total return: {results['Total Return (%)']:.2f}%")
        print(f"📉 Max drawdown: {results['Max Drawdown (%)']:.2f}%")
        print(f"🎯 Win rate: {results['Win Rate (%)']:.1f}%")
        print(f"⚡ Sharpe ratio: {results['Sharpe Ratio']:.2f}")
        
        # Show regime distribution
        regime_dist = data['Market_Regime'].value_counts()
        print(f"\n🎯 Market Regime Distribution:")
        for regime, count in regime_dist.items():
            pct = (count / len(data)) * 100
            print(f"  - {regime.capitalize()}: {count:,} periods ({pct:.1f}%)")
        
        # Show recent data sample
        print("\n📋 Recent Data Sample:")
        columns = ['Close', 'ADX', 'Market_Regime', 'Active_Strategy', 'Position']
        print(data[columns].tail(10))
        
        # Generate comprehensive charts
        print("\n📊 Generating comprehensive charts...")
        strategy.plot_comprehensive_results()
        
        # Ask user if they want to see individual strategy performance
        print("\n" + "="*60)
        user_input = input("Would you like to see individual strategy performance? (y/n): ")
        
        if user_input.lower() == 'y':
            run_individual_strategies(data_file)
        
    except FileNotFoundError:
        print(f"❌ Error: Data file '{data_file}' not found!")
        print("Make sure the CSV file is in the same directory.")
        print("Available files should include hourly BTC data.")
    except Exception as e:
        print(f"❌ Error running strategy: {e}")
        import traceback
        traceback.print_exc()

def run_individual_strategies(data_file):
    """Run individual strategies for comparison"""
    print("\n" + "="*60)
    print("📊 INDIVIDUAL STRATEGY PERFORMANCE COMPARISON")
    print("="*60)
    
    try:
        # Run Bollinger Band strategy
        print("\n1️⃣ Running Bollinger Band Mean Reversion Strategy...")
        bollinger = BollingerMeanReversionStrategy()
        bollinger_data, bollinger_results = bollinger.run_complete_analysis(data_file, initial_capital=10000)
        
        # Run EMA Crossover strategy
        print("\n2️⃣ Running EMA Crossover Strategy...")
        ema = EMACrossoverStrategy()
        ema_data, ema_results = ema.run_complete_analysis(data_file, initial_capital=10000)
        
        # Comparison summary
        print("\n" + "="*60)
        print("📊 STRATEGY COMPARISON SUMMARY")
        print("="*60)
        
        strategies = {
            'Bollinger Band': bollinger_results,
            'EMA Crossover': ema_results
        }
        
        metrics = ['Total Return (%)', 'Max Drawdown (%)', 'Win Rate (%)', 'Sharpe Ratio', 'Total Trades']
        
        print(f"{'Metric':<20} {'Bollinger':<12} {'EMA':<12}")
        print("-" * 50)
        
        for metric in metrics:
            bollinger_val = bollinger_results.get(metric, 0)
            ema_val = ema_results.get(metric, 0)
            
            if isinstance(bollinger_val, float):
                print(f"{metric:<20} {bollinger_val:>10.2f} {ema_val:>10.2f}")
            else:
                print(f"{metric:<20} {bollinger_val:>10} {ema_val:>10}")
        
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error running individual strategies: {e}")

def main():
    """Main function with menu options"""
    if len(sys.argv) > 1 and sys.argv[1] == '--individual':
        # Run individual strategies only
        data_file = "BTCUSD_Hourly_Ask_2024.01.01_2024.12.31.csv"
        run_individual_strategies(data_file)
    else:
        # Run hybrid strategy (default)
        run_hybrid_strategy()

if __name__ == "__main__":
    main()
