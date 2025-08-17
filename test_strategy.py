#!/usr/bin/env python3
"""
Quick test of the Market-Adaptive Hybrid Strategy
Shows key results without interactive plotting
"""

from market_adaptive_strategy import MarketAdaptiveStrategy

def main():
    print("🚀 Market-Adaptive Hybrid Strategy - Quick Test")
    print("=" * 60)
    
    # Initialize strategy
    strategy = MarketAdaptiveStrategy()
    
    # Run analysis without plotting
    data_file = "BTCUSD_Hourly_Ask_2024.01.01_2024.12.31.csv"
    
    try:
        # Load data and calculate indicators
        strategy.load_data(data_file)
        strategy.calculate_all_indicators()
        
        # Generate signals
        strategy.generate_hybrid_signals()
        
        # Run backtest
        results = strategy.backtest_strategy(initial_capital=10000)
        
        # Show results
        strategy.print_performance_summary()
        
        print("\n🎯 QUICK ANALYSIS SUMMARY:")
        print(f"📊 Total data points: {len(strategy.data):,}")
        print(f"💰 Final equity: ${results['Final Equity']:,.2f}")
        print(f"📈 Total return: {results['Total Return (%)']:.2f}%")
        print(f"🎯 Win rate: {results['Win Rate (%)']:.1f}%")
        print(f"📉 Max drawdown: {results['Max Drawdown (%)']:.2f}%")
        
        # Show regime distribution
        regime_dist = strategy.data['Market_Regime'].value_counts()
        print(f"\n📊 Market Regime Distribution:")
        for regime, count in regime_dist.items():
            pct = (count / len(strategy.data)) * 100
            print(f"  - {regime.capitalize()}: {count:,} periods ({pct:.1f}%)")
        
        # Show trading activity
        total_entries = (strategy.data['Position_Change'] == 1).sum()
        total_exits = (strategy.data['Position_Change'] == -1).sum()
        print(f"\n⚡ Trading Activity:")
        print(f"  - Total entries: {total_entries}")
        print(f"  - Total exits: {total_exits}")
        
        # Show strategy usage
        strategy_usage = strategy.data['Active_Strategy'].value_counts()
        print(f"\n🎯 Strategy Usage:")
        for strat, count in strategy_usage.items():
            pct = (count / len(strategy.data)) * 100
            print(f"  - {strat.capitalize()}: {count:,} periods ({pct:.1f}%)")
        
        print("\n✅ Market-Adaptive Strategy Test Completed Successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
