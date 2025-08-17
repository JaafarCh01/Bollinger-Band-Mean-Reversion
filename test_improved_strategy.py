#!/usr/bin/env python3
"""
Test the IMPROVED Market-Adaptive Strategy
Focus on validating whipsaw reduction and performance improvements
"""

from market_adaptive_strategy import MarketAdaptiveStrategy
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

def test_improved_strategy():
    print("🧪 Testing IMPROVED Market-Adaptive Strategy")
    print("=" * 60)
    print("🎯 Key Improvements in This Version:")
    print("  1. Regime Stability Filter:")
    print("     - Requires 3 consecutive bars of the same regime before trading")
    print("     - Prevents whipsaw from temporary regime changes")
    print("     - Minimum 5 bars in regime before allowing regime change exits")
    print("  2. Enhanced Bollinger Strategy:")
    print("     - Raised RSI oversold to 40 (better quality signals)")
    print("     - Added 50 EMA filter (avoid trading in downtrends)")
    print("     - Increased stop loss to 3% (fewer premature exits)")
    print("  3. Improved EMA Strategy:")
    print("     - Increased trailing stop to 4% (capture larger trends)")
    print("     - Increased stop loss to 3% (fewer false stops)")
    print("     - Extended time-based exit to 25 bars")
    print("  4. Better Risk Management:")
    print("     - Increased risk per trade to 2%")
    print("     - Increased max position size to 20% of equity")
    print("     - Less aggressive drawdown reduction")
    print("=" * 60)
    
    # Initialize improved strategy
    strategy = MarketAdaptiveStrategy(
        # Regime stability parameters
        regime_stability_bars=3, min_regime_duration=5,
        # Enhanced Bollinger parameters
        rsi_oversold=40, rsi_overbought=65,
        # Improved risk management
        risk_per_trade=0.02, stop_loss_pct=0.03, trailing_stop_pct=0.04
    )
    
    try:
        # Run analysis
        data, results = strategy.run_complete_analysis(
            'BTCUSD_Hourly_Ask_2024.01.01_2024.12.31.csv', 
            initial_capital=10000
        )
        
        print(f"\n🎉 IMPROVED STRATEGY RESULTS:")
        print(f"💰 Final Equity: ${results['Final Equity']:,.2f}")
        print(f"📈 Total Return: {results['Total Return (%)']:,.2f}%")
        print(f"🎯 Win Rate: {results['Win Rate (%)']:,.1f}%")
        print(f"📊 Total Trades: {results['Total Trades']}")
        print(f"⚡ Profit Factor: {results.get('Profit Factor', 'N/A')}")
        print(f"📉 Max Drawdown: {results['Max Drawdown (%)']:,.2f}%")
        
        # Analyze regime stability improvements
        print(f"\n📊 REGIME STABILITY ANALYSIS:")
        stable_regime_changes = 0
        raw_regime_changes = 0
        
        for i in range(1, len(data)):
            if data['Market_Regime'].iloc[i] != data['Market_Regime'].iloc[i-1]:
                raw_regime_changes += 1
            if 'Stable_Regime' in data.columns:
                if data['Stable_Regime'].iloc[i] != data['Stable_Regime'].iloc[i-1]:
                    stable_regime_changes += 1
        
        print(f"  - Raw regime changes: {raw_regime_changes}")
        print(f"  - Stable regime changes: {stable_regime_changes}")
        print(f"  - Whipsaw reduction: {((raw_regime_changes - stable_regime_changes) / raw_regime_changes * 100):.1f}%")
        
        # Count regime change exits
        regime_change_exits = 0
        if strategy.trade_log:
            for trade in strategy.trade_log:
                if trade.get('exit_reason') == 'regime_change':
                    regime_change_exits += 1
        
        print(f"  - Regime change exits: {regime_change_exits} (Target: <20)")
        
        # Show exit reason breakdown
        if strategy.trade_log:
            exit_reasons = {}
            for trade in strategy.trade_log:
                reason = trade.get('exit_reason', 'unknown')
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
            print(f"\n📊 EXIT REASON BREAKDOWN:")
            for reason, count in exit_reasons.items():
                pct = (count / len(strategy.trade_log)) * 100
                print(f"  - {reason.replace('_', ' ').title()}: {count} ({pct:.1f}%)")
        
        # Performance vs expected results
        print(f"\n🎯 PERFORMANCE VS EXPECTED RESULTS:")
        win_rate = results['Win Rate (%)']
        total_return = results['Total Return (%)']
        total_trades = results['Total Trades']
        
        print(f"  - Fewer Regime Change Exits: {regime_change_exits} (Expected: <20) {'✅' if regime_change_exits < 20 else '❌'}")
        print(f"  - Better Win Rate: {win_rate:.1f}% (Expected: >50%) {'✅' if win_rate > 50 else '❌'}")
        print(f"  - Higher Returns: {total_return:.1f}% (Expected: Positive) {'✅' if total_return > 0 else '❌'}")
        print(f"  - More Trades: {total_trades} (Expected: >35) {'✅' if total_trades > 35 else '❌'}")
        
        # Show strategy-specific performance
        if strategy.trade_log:
            bollinger_trades = [t for t in strategy.trade_log if t.get('strategy') == 'bollinger']
            ema_trades = [t for t in strategy.trade_log if t.get('strategy') == 'ema']
            
            if bollinger_trades:
                bollinger_wins = sum(1 for t in bollinger_trades if t['pnl'] > 0)
                bollinger_win_rate = (bollinger_wins / len(bollinger_trades)) * 100
                print(f"\n📊 BOLLINGER STRATEGY: {len(bollinger_trades)} trades, {bollinger_win_rate:.1f}% win rate")
            
            if ema_trades:
                ema_wins = sum(1 for t in ema_trades if t['pnl'] > 0)
                ema_win_rate = (ema_wins / len(ema_trades)) * 100
                print(f"📈 EMA STRATEGY: {len(ema_trades)} trades, {ema_win_rate:.1f}% win rate")
        
        print("\n✅ Improved Strategy Test Completed!")
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_improved_strategy()
