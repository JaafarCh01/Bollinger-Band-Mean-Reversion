#!/usr/bin/env python3
"""
Pure EMA Crossover Strategy
Based on the outstanding performance from the hybrid strategy: 71.8% win rate!

Entry Logic:
- Price > 55 EMA (uptrend confirmation)
- 21 EMA > 55 EMA (medium-term uptrend) 
- 5 EMA < 21 EMA (pullback condition)

Exit Logic:
- 5 EMA crosses above 21 EMA (natural exit - 95.2% win rate!)
- 4% trailing stop (after 2% profit)
- 3% hard stop loss
- 25 bars time-based exit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PureEMAStrategy:
    def __init__(self, 
                 # EMA Parameters (proven successful)
                 ema_short=5, ema_medium=21, ema_long=55,
                 # Risk Management (optimized from hybrid results)
                 risk_per_trade=0.02, stop_loss_pct=0.03, 
                 trailing_stop_pct=0.04, max_trades_per_day=5,
                 # Exit Parameters
                 time_based_exit_bars=25):
        """
        Initialize Pure EMA Strategy
        
        Parameters based on hybrid strategy's excellent EMA performance:
        - 71.8% win rate
        - 95.2% win rate on natural EMA crossover exits
        """
        self.ema_short = ema_short
        self.ema_medium = ema_medium
        self.ema_long = ema_long
        
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_trades_per_day = max_trades_per_day
        self.time_based_exit_bars = time_based_exit_bars
        
        self.data = None
        self.backtest_results = None
        self.trade_log = []
        self.current_trade = None
    
    def load_data(self, file_path):
        """Load and preprocess the CSV data"""
        print("Loading data for Pure EMA Strategy...")
        
        # Read CSV file
        self.data = pd.read_csv(file_path)
        
        # Parse datetime
        self.data['Time'] = pd.to_datetime(self.data['Time (EET)'])
        self.data.set_index('Time', inplace=True)
        
        # Drop the original time column and clean column names
        if 'Time (EET)' in self.data.columns:
            self.data.drop('Time (EET)', axis=1, inplace=True)
        
        # Clean column names (strip any whitespace)
        self.data.columns = self.data.columns.str.strip()
        
        # Ensure we have the expected columns
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if list(self.data.columns) != expected_columns:
            self.data.columns = expected_columns
        
        print(f"Data loaded successfully: {len(self.data)} rows")
        print(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")
        
        return self.data
    
    def calculate_emas(self):
        """Calculate the three EMAs"""
        print(f"Calculating EMAs: {self.ema_short}, {self.ema_medium}, {self.ema_long} periods")
        
        self.data['EMA_5'] = self.data['Close'].ewm(span=self.ema_short, adjust=False).mean()
        self.data['EMA_21'] = self.data['Close'].ewm(span=self.ema_medium, adjust=False).mean()
        self.data['EMA_55'] = self.data['Close'].ewm(span=self.ema_long, adjust=False).mean()
        
        print("EMAs calculated successfully!")
    
    def generate_signals(self):
        """Generate EMA crossover trading signals"""
        print("Generating Pure EMA signals...")
        
        # Initialize signal columns
        self.data['Signal'] = 0
        self.data['Position'] = 0
        
        # Entry conditions (proven from hybrid strategy)
        entry_condition = (
            (self.data['Close'] > self.data['EMA_55']) &      # Price above long-term trend
            (self.data['EMA_21'] > self.data['EMA_55']) &     # Medium-term above long-term
            (self.data['EMA_5'] < self.data['EMA_21'])        # Short-term below medium (pullback)
        )
        
        # Mark entry signals
        self.data.loc[entry_condition, 'Signal'] = 1
        
        # Track positions with enhanced exit logic
        self._track_positions()
        
        # Calculate position changes
        self.data['Position_Change'] = self.data['Position'].diff()
        
        entry_signals = (self.data['Position_Change'] == 1).sum()
        exit_signals = (self.data['Position_Change'] == -1).sum()
        
        print(f"Pure EMA signals generated: {entry_signals} entries, {exit_signals} exits")
        
        return self.data
    
    def _track_positions(self):
        """Track positions with proven exit logic from hybrid strategy"""
        position = 0
        entry_bar = None
        entry_price = None
        highest_price = None
        trailing_stop_price = None
        positions = []
        
        for i in range(len(self.data)):
            current_signal = self.data['Signal'].iloc[i]
            current_price = self.data['Close'].iloc[i]
            
            # Entry signal
            if current_signal == 1 and position == 0:
                position = 1
                entry_bar = i
                entry_price = current_price
                highest_price = current_price
                trailing_stop_price = None
                
                # Start trade log
                self.current_trade = {
                    'entry_bar': i,
                    'entry_price': current_price,
                    'entry_time': self.data.index[i],
                    'strategy': 'pure_ema',
                    'position_type': 'long'
                }
            
            # Update trailing stops and check exits
            elif position == 1:
                # Update highest price and trailing stop
                if current_price > highest_price:
                    highest_price = current_price
                    
                    # Activate trailing stop after 2% profit
                    if (current_price - entry_price) / entry_price >= 0.02:
                        trailing_stop_price = current_price * (1 - self.trailing_stop_pct)
                
                # Check exit conditions (in order of priority from hybrid results)
                should_exit = False
                exit_reason = None
                
                # 1. Natural EMA crossover exit (95.2% win rate!)
                if self.data['EMA_5'].iloc[i] > self.data['EMA_21'].iloc[i]:
                    should_exit = True
                    exit_reason = 'ema_crossover'
                
                # 2. Trailing stop (for profit protection)
                elif trailing_stop_price and current_price <= trailing_stop_price:
                    should_exit = True
                    exit_reason = 'trailing_stop'
                
                # 3. Hard stop loss (3%)
                elif current_price <= entry_price * (1 - self.stop_loss_pct):
                    should_exit = True
                    exit_reason = 'stop_loss'
                
                # 4. Time-based exit (25 bars)
                elif entry_bar is not None and (i - entry_bar) >= self.time_based_exit_bars:
                    should_exit = True
                    exit_reason = 'time_exit'
                
                if should_exit:
                    # Finalize trade
                    if self.current_trade:
                        self.current_trade['exit_bar'] = i
                        self.current_trade['exit_price'] = current_price
                        self.current_trade['exit_time'] = self.data.index[i]
                        self.current_trade['exit_reason'] = exit_reason
                        self.current_trade['bars_held'] = i - entry_bar
                        
                        # Calculate P&L
                        pnl_pct = (current_price - entry_price) / entry_price
                        self.current_trade['pnl'] = pnl_pct
                        self.current_trade['pnl_pct'] = pnl_pct * 100
                        
                        # Add to trade log
                        self.trade_log.append(self.current_trade.copy())
                        self.current_trade = None
                    
                    position = 0
                    entry_bar = None
                    entry_price = None
                    highest_price = None
                    trailing_stop_price = None
            
            positions.append(position)
        
        self.data['Position'] = positions
    
    def backtest_strategy(self, initial_capital=10000, transaction_cost=0.001):
        """Backtest the Pure EMA strategy"""
        print("Running Pure EMA backtest...")
        
        # Calculate returns
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Strategy_Returns'] = self.data['Position'].shift(1) * self.data['Returns']
        
        # Apply transaction costs
        position_changes = self.data['Position_Change'].abs()
        transaction_costs = position_changes * transaction_cost
        self.data['Strategy_Returns'] -= transaction_costs
        
        # Calculate cumulative returns
        self.data['Cumulative_Returns'] = (1 + self.data['Strategy_Returns']).cumprod()
        self.data['Cumulative_Market_Returns'] = (1 + self.data['Returns']).cumprod()
        
        # Calculate equity curves
        self.data['Equity'] = initial_capital * self.data['Cumulative_Returns']
        self.data['Market_Equity'] = initial_capital * self.data['Cumulative_Market_Returns']
        
        # Performance metrics
        total_return = (self.data['Cumulative_Returns'].iloc[-1] - 1) * 100
        market_return = (self.data['Cumulative_Market_Returns'].iloc[-1] - 1) * 100
        
        # Calculate drawdown
        rolling_max = self.data['Equity'].expanding().max()
        drawdown = (self.data['Equity'] - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        # Win rate and trade analysis from trade log
        if self.trade_log:
            winning_trades = sum(1 for trade in self.trade_log if trade['pnl'] > 0)
            total_trades = len(self.trade_log)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Profit factor
            gross_profit = sum(trade['pnl'] for trade in self.trade_log if trade['pnl'] > 0)
            gross_loss = abs(sum(trade['pnl'] for trade in self.trade_log if trade['pnl'] < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Average trade metrics
            avg_win = np.mean([trade['pnl'] for trade in self.trade_log if trade['pnl'] > 0]) * 100 if winning_trades > 0 else 0
            avg_loss = np.mean([trade['pnl'] for trade in self.trade_log if trade['pnl'] < 0]) * 100 if (total_trades - winning_trades) > 0 else 0
            avg_bars_held = np.mean([trade['bars_held'] for trade in self.trade_log])
            
        else:
            win_rate = 0
            total_trades = 0
            profit_factor = 0
            avg_win = 0
            avg_loss = 0
            avg_bars_held = 0
        
        # Sharpe ratio (annualized for hourly data)
        strategy_std = self.data['Strategy_Returns'].std() * np.sqrt(365 * 24)
        strategy_mean = self.data['Strategy_Returns'].mean() * 365 * 24
        sharpe_ratio = strategy_mean / strategy_std if strategy_std != 0 else 0
        
        self.backtest_results = {
            'Total Return (%)': total_return,
            'Market Return (%)': market_return,
            'Excess Return (%)': total_return - market_return,
            'Max Drawdown (%)': max_drawdown,
            'Win Rate (%)': win_rate,
            'Total Trades': total_trades,
            'Profit Factor': profit_factor,
            'Sharpe Ratio': sharpe_ratio,
            'Average Win (%)': avg_win,
            'Average Loss (%)': avg_loss,
            'Average Bars Held': avg_bars_held,
            'Final Equity': self.data['Equity'].iloc[-1]
        }
        
        print("Pure EMA backtest completed!")
        return self.backtest_results
    
    def print_performance_summary(self):
        """Print comprehensive performance summary"""
        if self.backtest_results is None:
            print("No backtest results available. Run backtest first.")
            return
        
        print("\\n" + "="*60)
        print("🚀 PURE EMA STRATEGY RESULTS")
        print("="*60)
        
        print("\\n📊 OVERALL PERFORMANCE:")
        for key, value in self.backtest_results.items():
            if isinstance(value, float):
                print(f"{key:<25}: {value:>10.2f}")
            else:
                print(f"{key:<25}: {value:>10}")
        
        # Show recent trades
        if self.trade_log:
            print(f"\\n📋 RECENT TRADES (Last 5):")
            recent_trades = self.trade_log[-5:] if len(self.trade_log) >= 5 else self.trade_log
            
            for i, trade in enumerate(recent_trades, 1):
                pnl_str = f"{trade['pnl_pct']:+.2f}%"
                status = "✅ WIN" if trade['pnl'] > 0 else "❌ LOSS"
                bars_held = trade['bars_held']
                print(f"  {i}. Entry: ${trade['entry_price']:.2f} → Exit: ${trade['exit_price']:.2f} "
                      f"({pnl_str}) {status} [{trade['exit_reason']}] ({bars_held} bars)")
            
            # Exit reason analysis
            exit_reasons = {}
            for trade in self.trade_log:
                reason = trade['exit_reason']
                if reason not in exit_reasons:
                    exit_reasons[reason] = {'count': 0, 'wins': 0, 'total_pnl': 0}
                exit_reasons[reason]['count'] += 1
                exit_reasons[reason]['total_pnl'] += trade['pnl']
                if trade['pnl'] > 0:
                    exit_reasons[reason]['wins'] += 1
            
            print(f"\\n📊 EXIT REASON ANALYSIS:")
            for reason, stats in exit_reasons.items():
                win_rate = (stats['wins'] / stats['count']) * 100 if stats['count'] > 0 else 0
                avg_return = (stats['total_pnl'] / stats['count']) * 100 if stats['count'] > 0 else 0
                print(f"  - {reason.replace('_', ' ').title()}: {stats['count']} trades "
                      f"({win_rate:.1f}% win rate, {avg_return:+.2f}% avg return)")
        
        print("="*60)
    
    def plot_results(self, start_date=None, end_date=None, figsize=(15, 12)):
        """Plot Pure EMA strategy results"""
        if self.data is None:
            print("No data available. Load data first.")
            return
        
        # Filter data for plotting if dates specified
        plot_data = self.data.copy()
        if start_date:
            plot_data = plot_data[plot_data.index >= start_date]
        if end_date:
            plot_data = plot_data[plot_data.index <= end_date]
        
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        fig.suptitle('Pure EMA Strategy Analysis', fontsize=16, fontweight='bold')
        
        # 1. Price and EMAs with signals
        axes[0].plot(plot_data.index, plot_data['Close'], label='BTC Price', color='black', linewidth=1.5)
        axes[0].plot(plot_data.index, plot_data['EMA_5'], label='EMA 5', color='red', alpha=0.8)
        axes[0].plot(plot_data.index, plot_data['EMA_21'], label='EMA 21', color='blue', alpha=0.8)
        axes[0].plot(plot_data.index, plot_data['EMA_55'], label='EMA 55', color='green', alpha=0.8)
        
        # Mark signals
        buy_signals = plot_data[plot_data['Position_Change'] == 1]
        exit_signals = plot_data[plot_data['Position_Change'] == -1]
        
        axes[0].scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', 
                       s=50, label='Entry Signal', zorder=5)
        axes[0].scatter(exit_signals.index, exit_signals['Close'], color='red', marker='v', 
                       s=50, label='Exit Signal', zorder=5)
        
        axes[0].set_title('BTC Price with EMA Lines and Trading Signals')
        axes[0].set_ylabel('Price (USD)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Position tracking
        axes[1].plot(plot_data.index, plot_data['Position'], label='Position', 
                    color='orange', linewidth=2, drawstyle='steps-post')
        axes[1].fill_between(plot_data.index, 0, plot_data['Position'], alpha=0.3, 
                           color='orange', step='post')
        axes[1].set_title('Trading Position (1=Long, 0=Neutral)')
        axes[1].set_ylabel('Position')
        axes[1].set_ylim(-0.5, 1.5)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Equity curve comparison
        axes[2].plot(plot_data.index, plot_data['Equity'], label='Pure EMA Strategy', 
                    color='green', linewidth=2.5)
        axes[2].plot(plot_data.index, plot_data['Market_Equity'], label='Buy & Hold', 
                    color='blue', linewidth=2, alpha=0.7)
        axes[2].set_title('Equity Curve Comparison')
        axes[2].set_ylabel('Equity (USD)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self, file_path, initial_capital=10000):
        """Run complete Pure EMA strategy analysis"""
        print("🚀 Starting Pure EMA Strategy Analysis...")
        print("=" * 60)
        print("📊 Based on hybrid strategy's outstanding EMA performance:")
        print("   - 71.8% win rate")
        print("   - 95.2% win rate on natural EMA crossover exits")
        print("   - No regime switching interference!")
        print("=" * 60)
        
        # Load data
        self.load_data(file_path)
        
        # Calculate EMAs
        self.calculate_emas()
        
        # Generate signals
        self.generate_signals()
        
        # Run backtest
        self.backtest_strategy(initial_capital=initial_capital)
        
        # Print results
        self.print_performance_summary()
        
        return self.data, self.backtest_results

# Example usage
if __name__ == "__main__":
    # Initialize Pure EMA strategy
    strategy = PureEMAStrategy()
    
    # Run complete analysis
    data_file = "BTCUSD_Hourly_Ask_2024.01.01_2024.12.31.csv"
    data, results = strategy.run_complete_analysis(data_file, initial_capital=10000)
    
    # Plot results
    strategy.plot_results()
