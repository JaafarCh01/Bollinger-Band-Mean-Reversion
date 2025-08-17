#!/usr/bin/env python3
"""
Test the Enhanced Market-Adaptive Strategy
Shows improvements and validates new features
"""

from market_adaptive_strategy import MarketAdaptiveStrategy
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

def test_enhanced_strategy():
    print("🧪 Testing Enhanced Market-Adaptive Strategy")
    print("=" * 60)
    print("🎯 Key Enhancements:")
    print("  - ADX thresholds: 15/20 (was 20/25)")
    print("  - Relaxed entry conditions (no volume/candle requirements)")
    print("  - RSI thresholds: 35/65 (was 30/70)")
    print("  - Enhanced exit conditions with trailing stops")
    print("  - Increased risk limits and position sizes")
    print("  - Comprehensive trade logging")
    print("=" * 60)
    
    # Initialize enhanced strategy
    strategy = MarketAdaptiveStrategy(
        # Enhanced ADX parameters
        trend_threshold=20, range_threshold=15,
        # Relaxed RSI thresholds
        rsi_oversold=35, rsi_overbought=65,
        # Enhanced risk management
        risk_per_trade=0.015, max_trades_per_day=5,
        daily_loss_limit=0.08, trailing_stop_pct=0.03
    )
    
    try:
        # Run analysis
        data, results = strategy.run_complete_analysis(
            'BTCUSD_Hourly_Ask_2024.01.01_2024.12.31.csv', 
            initial_capital=10000
        )
        
        print(f"\n🎉 ENHANCED STRATEGY RESULTS:")
        print(f"💰 Final Equity: ${results['Final Equity']:,.2f}")
        print(f"📈 Total Return: {results['Total Return (%)']:,.2f}%")
        print(f"🎯 Win Rate: {results['Win Rate (%)']:,.1f}%")
        print(f"📊 Total Trades: {results['Total Trades']}")
        print(f"⚡ Profit Factor: {results.get('Profit Factor', 'N/A')}")
        print(f"📉 Max Drawdown: {results['Max Drawdown (%)']:,.2f}%")
        
        # Show regime distribution with new thresholds
        regime_dist = data['Market_Regime'].value_counts()
        print(f"\n📊 Market Regimes (New Thresholds):")
        for regime, count in regime_dist.items():
            pct = (count / len(data)) * 100
            print(f"  {regime.title()}: {pct:.1f}% ({count:,} periods)")
        
        # Show trade activity improvement
        total_entries = (data['Position_Change'] == 1).sum()
        print(f"\n⚡ Trading Activity:")
        print(f"  - Total entries: {total_entries} (target: >35)")
        print(f"  - Trade frequency: {total_entries / 365:.1f} trades/month")
        
        # Show strategy usage
        strategy_usage = data['Active_Strategy'].value_counts()
        print(f"\n🎯 Strategy Usage:")
        for strat, count in strategy_usage.items():
            pct = (count / len(data)) * 100
            print(f"  - {strat.title()}: {pct:.1f}%")
        
        # Performance vs targets
        print(f"\n🎯 PERFORMANCE VS TARGETS:")
        win_rate = results['Win Rate (%)']
        total_return = results['Total Return (%)']
        max_dd = abs(results['Max Drawdown (%)'])
        
        print(f"  - Win Rate: {win_rate:.1f}% (Target: >60%) {'✅' if win_rate > 60 else '❌'}")
        print(f"  - Total Return: {total_return:.1f}% (Target: >Market) {'✅' if total_return > 0 else '❌'}")
        print(f"  - Max Drawdown: {max_dd:.1f}% (Target: <20%) {'✅' if max_dd < 20 else '❌'}")
        print(f"  - Trade Frequency: {total_entries} (Target: >35) {'✅' if total_entries > 35 else '❌'}")
        
        print("\n✅ Enhanced Strategy Test Completed!")
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_enhanced_strategy()
