#!/usr/bin/env python3
"""
Bollinger Band Mean Reversion Strategy Implementation
Author: Trading Strategy Implementation
Date: 2025

This script implements a comprehensive Bollinger Band mean reversion strategy
for cryptocurrency trading with backtesting and performance analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BollingerMeanReversionStrategy:
    def __init__(self, bb_period=20, bb_std_dev=2, rsi_period=14, 
                 rsi_oversold=30, rsi_overbought=70):
        """
        Initialize the Bollinger Band Mean Reversion Strategy
        
        Parameters:
        -----------
        bb_period : int, default=20
            Period for Bollinger Bands calculation
        bb_std_dev : float, default=2
            Standard deviations for Bollinger Bands
        rsi_period : int, default=14
            Period for RSI calculation
        rsi_oversold : float, default=30
            RSI oversold threshold
        rsi_overbought : float, default=70
            RSI overbought threshold
        """
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        
        self.data = None
        self.signals = None
        self.backtest_results = None
    
    def load_data(self, file_path):
        """Load and preprocess the CSV data"""
        print("Loading data...")
        
        # Read CSV file
        self.data = pd.read_csv(file_path)
        
        # Parse datetime
        self.data['Time'] = pd.to_datetime(self.data['Time (EET)'])
        self.data.set_index('Time', inplace=True)
        
        # Clean column names
        self.data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        print(f"Data loaded successfully: {len(self.data)} rows")
        print(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")
        
        return self.data
    
    def calculate_bollinger_bands(self):
        """Calculate Bollinger Bands"""
        # Middle Band (SMA)
        self.data['BB_Middle'] = self.data['Close'].rolling(window=self.bb_period).mean()
        
        # Standard Deviation
        rolling_std = self.data['Close'].rolling(window=self.bb_period).std()
        
        # Upper and Lower Bands
        self.data['BB_Upper'] = self.data['BB_Middle'] + (self.bb_std_dev * rolling_std)
        self.data['BB_Lower'] = self.data['BB_Middle'] - (self.bb_std_dev * rolling_std)
        
        # Bollinger Band Position (0 = lower band, 1 = upper band)
        self.data['BB_Position'] = (self.data['Close'] - self.data['BB_Lower']) / (self.data['BB_Upper'] - self.data['BB_Lower'])
        
        print("Bollinger Bands calculated")
    
    def calculate_rsi(self):
        """Calculate Relative Strength Index (RSI)"""
        delta = self.data['Close'].diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        print("RSI calculated")
    
    def generate_signals(self):
        """Generate trading signals based on Bollinger Bands and RSI"""
        print("Generating trading signals...")
        
        # Initialize signal columns
        self.data['Signal'] = 0
        self.data['Position'] = 0
        
        # Buy signals: Price touches lower Bollinger Band AND RSI is oversold
        buy_condition = (
            (self.data['Close'] <= self.data['BB_Lower']) & 
            (self.data['RSI'] <= self.rsi_oversold)
        )
        
        # Sell signals: Price touches upper Bollinger Band AND RSI is overbought
        sell_condition = (
            (self.data['Close'] >= self.data['BB_Upper']) & 
            (self.data['RSI'] >= self.rsi_overbought)
        )
        
        # Exit signals: Price crosses back to middle band
        exit_long_condition = self.data['Close'] >= self.data['BB_Middle']
        exit_short_condition = self.data['Close'] <= self.data['BB_Middle']
        
        # Generate signals
        self.data.loc[buy_condition, 'Signal'] = 1  # Buy signal
        self.data.loc[sell_condition, 'Signal'] = -1  # Sell signal
        
        # Track positions
        position = 0
        positions = []
        
        for i in range(len(self.data)):
            current_signal = self.data['Signal'].iloc[i]
            current_close = self.data['Close'].iloc[i]
            current_bb_middle = self.data['BB_Middle'].iloc[i]
            
            # Entry signals
            if current_signal == 1 and position == 0:  # Buy signal
                position = 1
            elif current_signal == -1 and position == 0:  # Sell signal
                position = -1
            
            # Exit conditions
            elif position == 1 and current_close >= current_bb_middle:  # Exit long
                position = 0
            elif position == -1 and current_close <= current_bb_middle:  # Exit short
                position = 0
            
            positions.append(position)
        
        self.data['Position'] = positions
        
        # Calculate position changes for trade identification
        self.data['Position_Change'] = self.data['Position'].diff()
        
        buy_signals = (self.data['Position_Change'] == 1).sum()
        sell_signals = (self.data['Position_Change'] == -1).sum()
        
        print(f"Signals generated: {buy_signals} buy signals, {sell_signals} sell signals")
        
        return self.data
    
    def backtest_strategy(self, initial_capital=10000, transaction_cost=0.001):
        """
        Backtest the strategy
        
        Parameters:
        -----------
        initial_capital : float, default=10000
            Initial capital for backtesting
        transaction_cost : float, default=0.001
            Transaction cost as percentage (0.1%)
        """
        print("Running backtest...")
        
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Strategy_Returns'] = self.data['Position'].shift(1) * self.data['Returns']
        
        # Apply transaction costs
        position_changes = self.data['Position_Change'].abs()
        transaction_costs = position_changes * transaction_cost
        self.data['Strategy_Returns'] -= transaction_costs
        
        # Calculate cumulative returns
        self.data['Cumulative_Returns'] = (1 + self.data['Strategy_Returns']).cumprod()
        self.data['Cumulative_Market_Returns'] = (1 + self.data['Returns']).cumprod()
        
        # Calculate equity curve
        self.data['Equity'] = initial_capital * self.data['Cumulative_Returns']
        self.data['Market_Equity'] = initial_capital * self.data['Cumulative_Market_Returns']
        
        # Performance metrics
        total_return = (self.data['Cumulative_Returns'].iloc[-1] - 1) * 100
        market_return = (self.data['Cumulative_Market_Returns'].iloc[-1] - 1) * 100
        
        # Calculate drawdown
        rolling_max = self.data['Equity'].expanding().max()
        drawdown = (self.data['Equity'] - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        # Win rate calculation
        trades = self.data[self.data['Position_Change'] != 0].copy()
        if len(trades) > 1:
            trade_returns = []
            entry_price = None
            entry_position = None
            
            for idx, row in trades.iterrows():
                if entry_price is None:  # Entry
                    entry_price = row['Close']
                    entry_position = row['Position']
                else:  # Exit
                    if entry_position == 1:  # Long trade
                        trade_return = (row['Close'] - entry_price) / entry_price
                    else:  # Short trade
                        trade_return = (entry_price - row['Close']) / entry_price
                    
                    trade_returns.append(trade_return)
                    entry_price = None
                    entry_position = None
            
            winning_trades = sum(1 for ret in trade_returns if ret > 0)
            total_trades = len(trade_returns)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        else:
            win_rate = 0
            total_trades = 0
        
        # Sharpe ratio (annualized)
        strategy_std = self.data['Strategy_Returns'].std() * np.sqrt(365 * 24 * 4)  # 15-min data
        strategy_mean = self.data['Strategy_Returns'].mean() * 365 * 24 * 4
        sharpe_ratio = strategy_mean / strategy_std if strategy_std != 0 else 0
        
        self.backtest_results = {
            'Total Return (%)': total_return,
            'Market Return (%)': market_return,
            'Excess Return (%)': total_return - market_return,
            'Max Drawdown (%)': max_drawdown,
            'Win Rate (%)': win_rate,
            'Total Trades': total_trades,
            'Sharpe Ratio': sharpe_ratio,
            'Final Equity': self.data['Equity'].iloc[-1]
        }
        
        print("Backtest completed!")
        return self.backtest_results
    
    def print_performance_summary(self):
        """Print performance summary"""
        if self.backtest_results is None:
            print("No backtest results available. Run backtest first.")
            return
        
        print("\n" + "="*50)
        print("BOLLINGER BAND MEAN REVERSION STRATEGY RESULTS")
        print("="*50)
        
        for key, value in self.backtest_results.items():
            if isinstance(value, float):
                print(f"{key:<25}: {value:>10.2f}")
            else:
                print(f"{key:<25}: {value:>10}")
        
        print("="*50)
    
    def plot_results(self, start_date=None, end_date=None, figsize=(15, 12)):
        """
        Plot comprehensive strategy results
        
        Parameters:
        -----------
        start_date : str, optional
            Start date for plotting (YYYY-MM-DD format)
        end_date : str, optional
            End date for plotting (YYYY-MM-DD format)
        figsize : tuple, default=(15, 12)
            Figure size for plots
        """
        if self.data is None:
            print("No data available. Load data first.")
            return
        
        # Filter data for plotting if dates specified
        plot_data = self.data.copy()
        if start_date:
            plot_data = plot_data[plot_data.index >= start_date]
        if end_date:
            plot_data = plot_data[plot_data.index <= end_date]
        
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        fig.suptitle('Bollinger Band Mean Reversion Strategy Analysis', fontsize=16, fontweight='bold')
        
        # 1. Price and Bollinger Bands
        axes[0].plot(plot_data.index, plot_data['Close'], label='BTC Price', color='black', linewidth=1)
        axes[0].plot(plot_data.index, plot_data['BB_Upper'], label='Upper Band', color='red', alpha=0.7)
        axes[0].plot(plot_data.index, plot_data['BB_Middle'], label='Middle Band (SMA20)', color='blue', alpha=0.7)
        axes[0].plot(plot_data.index, plot_data['BB_Lower'], label='Lower Band', color='green', alpha=0.7)
        axes[0].fill_between(plot_data.index, plot_data['BB_Upper'], plot_data['BB_Lower'], alpha=0.1, color='gray')
        
        # Mark buy/sell signals
        buy_signals = plot_data[plot_data['Position_Change'] == 1]
        sell_signals = plot_data[plot_data['Position_Change'] == -1]
        exit_long_signals = plot_data[(plot_data['Position_Change'] == -1) & (plot_data['Position'].shift(1) == 1)]
        exit_short_signals = plot_data[(plot_data['Position_Change'] == 1) & (plot_data['Position'].shift(1) == -1)]
        
        axes[0].scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', s=50, label='Buy Signal', zorder=5)
        axes[0].scatter(sell_signals.index, sell_signals['Close'], color='red', marker='v', s=50, label='Sell Signal', zorder=5)
        
        axes[0].set_title('BTC Price with Bollinger Bands and Trading Signals')
        axes[0].set_ylabel('Price (USD)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. RSI
        axes[1].plot(plot_data.index, plot_data['RSI'], label='RSI', color='purple', linewidth=1)
        axes[1].axhline(y=self.rsi_overbought, color='red', linestyle='--', alpha=0.7, label=f'Overbought ({self.rsi_overbought})')
        axes[1].axhline(y=self.rsi_oversold, color='green', linestyle='--', alpha=0.7, label=f'Oversold ({self.rsi_oversold})')
        axes[1].axhline(y=50, color='gray', linestyle='-', alpha=0.5)
        axes[1].fill_between(plot_data.index, self.rsi_overbought, 100, alpha=0.1, color='red')
        axes[1].fill_between(plot_data.index, 0, self.rsi_oversold, alpha=0.1, color='green')
        
        axes[1].set_title('Relative Strength Index (RSI)')
        axes[1].set_ylabel('RSI')
        axes[1].set_ylim(0, 100)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Position
        axes[2].plot(plot_data.index, plot_data['Position'], label='Position', color='orange', linewidth=2)
        axes[2].fill_between(plot_data.index, 0, plot_data['Position'], alpha=0.3, color='orange')
        axes[2].set_title('Trading Position (1=Long, -1=Short, 0=Neutral)')
        axes[2].set_ylabel('Position')
        axes[2].set_ylim(-1.5, 1.5)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 4. Equity Curve
        axes[3].plot(plot_data.index, plot_data['Equity'], label='Strategy Equity', color='green', linewidth=2)
        axes[3].plot(plot_data.index, plot_data['Market_Equity'], label='Buy & Hold', color='blue', linewidth=2, alpha=0.7)
        axes[3].set_title('Equity Curve Comparison')
        axes[3].set_ylabel('Equity (USD)')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self, file_path, initial_capital=10000):
        """Run complete strategy analysis"""
        print("Starting Bollinger Band Mean Reversion Strategy Analysis...")
        print("-" * 60)
        
        # Load data
        self.load_data(file_path)
        
        # Calculate indicators
        self.calculate_bollinger_bands()
        self.calculate_rsi()
        
        # Generate signals
        self.generate_signals()
        
        # Run backtest
        self.backtest_strategy(initial_capital=initial_capital)
        
        # Print results
        self.print_performance_summary()
        
        return self.data, self.backtest_results

# Example usage
if __name__ == "__main__":
    # Initialize strategy
    strategy = BollingerMeanReversionStrategy()
    
    # Run complete analysis
    data_file = "BTCUSD_15 Mins_Ask_2025.01.01_2025.08.16.csv"
    data, results = strategy.run_complete_analysis(data_file, initial_capital=10000)
    
    # Plot results
    strategy.plot_results()
