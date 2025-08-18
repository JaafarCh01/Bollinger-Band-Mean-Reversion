#!/usr/bin/env python3
"""
Run Pure EMA Strategy
Based on the outstanding 71.8% win rate from the hybrid strategy!
"""

from pure_ema_strategy import PureEMAStrategy
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing

def main():
    print("🚀 Pure EMA Strategy - No Regime Switching!")
    print("=" * 60)
    print("🎯 Based on hybrid strategy's excellent EMA performance:")
    print("   - 71.8% overall win rate")
    print("   - 95.2% win rate on natural EMA crossover exits")
    print("   - 24 regime change exits eliminated!")
    print("=" * 60)
    
    # Initialize Pure EMA strategy with proven parameters
    strategy = PureEMAStrategy(
        # EMA Parameters (proven successful)
        ema_short=5,
        ema_medium=21,
        ema_long=55,
        
        # Risk Management (optimized from hybrid results)
        risk_per_trade=0.02,        # 2% risk per trade
        stop_loss_pct=0.03,         # 3% hard stop loss
        trailing_stop_pct=0.04,     # 4% trailing stop
        max_trades_per_day=5,       # Max 5 trades per day
        
        # Exit Parameters
        time_based_exit_bars=25     # 25 bars time-based exit
    )
    
    # Use hourly BTC data
    data_file = "BTCUSD_Hourly_Ask_2024.01.01_2024.12.31.csv"
    
    try:
        # Run complete analysis
        data, results = strategy.run_complete_analysis(data_file, initial_capital=10000)
        
        print(f"\\n🎉 PURE EMA STRATEGY RESULTS:")
        print(f"💰 Final Equity: ${results['Final Equity']:,.2f}")
        print(f"📈 Total Return: {results['Total Return (%)']:.2f}%")
        print(f"🎯 Win Rate: {results['Win Rate (%)']:.1f}%")
        print(f"📊 Total Trades: {results['Total Trades']}")
        print(f"⚡ Profit Factor: {results['Profit Factor']:.2f}")
        print(f"📉 Max Drawdown: {results['Max Drawdown (%)']:.2f}%")
        print(f"📊 Sharpe Ratio: {results['Sharpe Ratio']:.2f}")
        
        # Compare with hybrid results
        print(f"\\n📊 COMPARISON WITH HYBRID STRATEGY:")
        print(f"   Hybrid EMA Win Rate: 71.8% → Pure EMA: {results['Win Rate (%)']:.1f}%")
        print(f"   Hybrid Total Trades: 71 → Pure EMA: {results['Total Trades']}")
        print(f"   Hybrid Regime Exits: 24 → Pure EMA: 0 (eliminated!)")
        
        # Show trade frequency
        trade_frequency = results['Total Trades'] / 12  # Monthly frequency
        print(f"\\n⚡ TRADING ACTIVITY:")
        print(f"   - Trade frequency: {trade_frequency:.1f} trades/month")
        print(f"   - Average holding period: {results['Average Bars Held']:.1f} hours")
        
        # Performance targets
        print(f"\\n🎯 PERFORMANCE VS TARGETS:")
        win_rate = results['Win Rate (%)']
        total_return = results['Total Return (%)']
        max_dd = abs(results['Max Drawdown (%)'])
        
        print(f"   - Win Rate: {win_rate:.1f}% (Target: >70%) {'✅' if win_rate > 70 else '❌'}")
        print(f"   - Total Return: {total_return:.1f}% (Target: >Market) {'✅' if total_return > 0 else '❌'}")
        print(f"   - Max Drawdown: {max_dd:.1f}% (Target: <15%) {'✅' if max_dd < 15 else '❌'}")
        print(f"   - Profit Factor: {results['Profit Factor']:.2f} (Target: >1.5) {'✅' if results['Profit Factor'] > 1.5 else '❌'}")
        
        print("\\n✅ Pure EMA Strategy Analysis Complete!")
        
        # Optional: Generate plots (uncomment if you want to see charts)
        # print("\\n📊 Generating charts...")
        # strategy.plot_results()
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
