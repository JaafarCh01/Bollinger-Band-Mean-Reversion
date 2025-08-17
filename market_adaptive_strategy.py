#!/usr/bin/env python3
"""
Market-Adaptive Hybrid Trading Strategy
Author: Trading Strategy Implementation
Date: 2025

This script implements the Market-Adaptive Hybrid Strategy that automatically
switches between Bollinger Band Mean Reversion (range markets) and EMA Crossover
(trending markets) based on ADX market regime detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from regime_detector import MarketRegimeDetector
from bollinger_strategy import BollingerMeanReversionStrategy
from ema_strategy import EMACrossoverStrategy

class MarketAdaptiveStrategy:
    def __init__(self, 
                 # ADX Parameters (Updated for crypto volatility)
                 adx_period=14, trend_threshold=20, range_threshold=15,
                 # Bollinger Band Parameters (Relaxed thresholds)
                 bb_period=20, bb_std_dev=2, rsi_period=14, 
                 rsi_oversold=35, rsi_overbought=65,
                 # EMA Parameters
                 ema_short=5, ema_medium=21, ema_long=55,
                 # Risk Management (IMPROVED limits)
                 risk_per_trade=0.02, stop_loss_pct=0.03, 
                 max_trades_per_day=5, daily_loss_limit=0.08,
                 # Enhanced trailing stop parameter
                 trailing_stop_pct=0.04,
                 # Regime stability parameters
                 regime_stability_bars=3, min_regime_duration=5):
        """
        Initialize Market-Adaptive Hybrid Strategy
        
        Parameters:
        -----------
        adx_period : int, default=14
            Period for ADX calculation
        trend_threshold : float, default=25
            ADX threshold for trending market
        range_threshold : float, default=20
            ADX threshold for range-bound market
        bb_period : int, default=20
            Bollinger Band period
        bb_std_dev : float, default=2
            Bollinger Band standard deviations
        rsi_period : int, default=14
            RSI period
        rsi_oversold : float, default=30
            RSI oversold threshold
        rsi_overbought : float, default=70
            RSI overbought threshold
        ema_short : int, default=5
            Short EMA period
        ema_medium : int, default=21
            Medium EMA period
        ema_long : int, default=55
            Long EMA period
        risk_per_trade : float, default=0.01
            Risk percentage per trade
        stop_loss_pct : float, default=0.02
            Stop loss percentage
        max_trades_per_day : int, default=3
            Maximum trades per day
        daily_loss_limit : float, default=0.05
            Daily loss limit percentage
        """
        # Initialize components
        self.regime_detector = MarketRegimeDetector(adx_period, trend_threshold, range_threshold)
        self.bollinger_strategy = BollingerMeanReversionStrategy(
            bb_period, bb_std_dev, rsi_period, rsi_oversold, rsi_overbought)
        self.ema_strategy = EMACrossoverStrategy(ema_short, ema_medium, ema_long)
        
        # Risk management parameters
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.max_trades_per_day = max_trades_per_day
        self.daily_loss_limit = daily_loss_limit
        self.trailing_stop_pct = trailing_stop_pct
        
        # Regime stability parameters
        self.regime_stability_bars = regime_stability_bars
        self.min_regime_duration = min_regime_duration
        
        # Strategy state
        self.data = None
        self.backtest_results = None
        self.regime_stats = {
            'trending': {'trades': 0, 'wins': 0, 'total_return': 0},
            'range': {'trades': 0, 'wins': 0, 'total_return': 0}
        }
        
        # Trade logging
        self.trade_log = []
        self.current_trade = None
        
        # Regime tracking for stability
        self.regime_history = []
        self.current_stable_regime = None
        self.regime_start_bar = None
    
    def load_data(self, file_path):
        """Load and preprocess the CSV data"""
        print("Loading data for Market-Adaptive Strategy...")
        
        # Read CSV file
        self.data = pd.read_csv(file_path)
        
        # Parse datetime
        self.data['Time'] = pd.to_datetime(self.data['Time (EET)'])
        self.data.set_index('Time', inplace=True)
        
        # Drop the original time column and clean column names
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
    
    def calculate_all_indicators(self):
        """Calculate all technical indicators needed for both strategies"""
        print("Calculating all technical indicators...")
        
        # Market regime detection (ADX)
        self.data = self.regime_detector.run_regime_analysis(self.data)
        
        # Bollinger Bands and RSI
        self.bollinger_strategy.data = self.data.copy()
        self.bollinger_strategy.calculate_bollinger_bands()
        self.bollinger_strategy.calculate_rsi()
        
        # EMAs
        self.ema_strategy.data = self.data.copy()
        self.ema_strategy.calculate_emas()
        
        # Add 50 EMA for Bollinger strategy filter
        self.data['EMA_50'] = self.data['Close'].ewm(span=50, adjust=False).mean()
        
        # Copy calculated indicators back to main data
        bollinger_cols = ['BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Position', 'RSI']
        ema_cols = ['EMA_5', 'EMA_21', 'EMA_55']
        
        for col in bollinger_cols:
            if col in self.bollinger_strategy.data.columns:
                self.data[col] = self.bollinger_strategy.data[col]
        
        for col in ema_cols:
            if col in self.ema_strategy.data.columns:
                self.data[col] = self.ema_strategy.data[col]
        
        print("All indicators calculated successfully!")
    
    def _get_stable_regime(self, i):
        """Get stable regime with stability filter"""
        if i < self.regime_stability_bars:
            return None
        
        current_regime = self.data['Market_Regime'].iloc[i]
        
        # Check if last N bars have same regime
        recent_regimes = [self.data['Market_Regime'].iloc[j] for j in range(i - self.regime_stability_bars + 1, i + 1)]
        
        if all(r == current_regime for r in recent_regimes):
            # Check if we're changing from a different stable regime
            if (self.current_stable_regime != current_regime):
                # If we have a position and regime is changing, need minimum duration
                if (self.current_stable_regime is not None and 
                    self.regime_start_bar is not None and 
                    (i - self.regime_start_bar) < self.min_regime_duration):
                    return self.current_stable_regime  # Stay in current regime
                
                # Update to new stable regime
                self.current_stable_regime = current_regime
                self.regime_start_bar = i
            
            return current_regime
        
        # No stable regime detected, maintain current if exists
        return self.current_stable_regime
    
    def generate_hybrid_signals(self):
        """Generate trading signals based on stable market regime"""
        print("Generating Market-Adaptive signals with regime stability filter...")
        
        # Initialize signal columns
        self.data['Signal'] = 0
        self.data['Position'] = 0
        self.data['Active_Strategy'] = 'none'
        self.data['Signal_Source'] = 'none'
        self.data['Stable_Regime'] = 'none'
        
        # Calculate additional conditions (kept for compatibility but not used in relaxed conditions)
        self.data['Bullish_Candle'] = self.data['Close'] > self.data['Open']
        self.data['Volume_Spike'] = self.data['Volume'] > (self.data['Volume'].rolling(5).mean() * 1.1)
        
        # Generate regime-specific signals with stability filter
        for i in range(len(self.data)):
            stable_regime = self._get_stable_regime(i)
            self.data.loc[self.data.index[i], 'Stable_Regime'] = stable_regime or 'none'
            
            if stable_regime == 'trending':
                # Use EMA Crossover strategy
                signal = self._get_ema_signal(i)
                self.data.loc[self.data.index[i], 'Signal'] = signal
                self.data.loc[self.data.index[i], 'Active_Strategy'] = 'ema'
                if signal != 0:
                    self.data.loc[self.data.index[i], 'Signal_Source'] = 'ema_crossover'
                    
            elif stable_regime == 'range':
                # Use Enhanced Bollinger Band strategy
                signal = self._get_enhanced_bollinger_signal(i)
                self.data.loc[self.data.index[i], 'Signal'] = signal
                self.data.loc[self.data.index[i], 'Active_Strategy'] = 'bollinger'
                if signal != 0:
                    self.data.loc[self.data.index[i], 'Signal_Source'] = 'bollinger_mean_reversion'
            
            # No stable regime or transition: no trading
            else:
                self.data.loc[self.data.index[i], 'Active_Strategy'] = 'transition'
        
        # Track positions with regime switching logic
        self._track_positions_with_regime_switching()
        
        # Calculate position changes for trade identification
        self.data['Position_Change'] = self.data['Position'].diff()
        
        # Count signals by strategy
        ema_signals = len(self.data[(self.data['Signal_Source'] == 'ema_crossover') & (self.data['Signal'] == 1)])
        bollinger_signals = len(self.data[(self.data['Signal_Source'] == 'bollinger_mean_reversion') & (self.data['Signal'] != 0)])
        total_entries = (self.data['Position_Change'] == 1).sum()
        
        print(f"Hybrid signals generated:")
        print(f"- EMA Crossover entries: {ema_signals}")
        print(f"- Bollinger Band entries: {bollinger_signals}")
        print(f"- Total position entries: {total_entries}")
        
        return self.data
    
    def _get_ema_signal(self, i):
        """Get EMA crossover signal for index i (relaxed conditions)"""
        try:
            # Entry conditions for EMA strategy (long only) - removed bullish candle requirement
            if (self.data['Close'].iloc[i] > self.data['EMA_55'].iloc[i] and
                self.data['EMA_21'].iloc[i] > self.data['EMA_55'].iloc[i] and
                self.data['EMA_5'].iloc[i] < self.data['EMA_21'].iloc[i]):
                return 1  # Buy signal
            
            return 0
        except (IndexError, KeyError):
            return 0
    
    def _get_enhanced_bollinger_signal(self, i):
        """Get Enhanced Bollinger Band signal with 50 EMA filter and RSI 40"""
        try:
            # Enhanced long entry conditions:
            # 1. Price below lower band
            # 2. RSI oversold (raised to 40 for better quality)
            # 3. Price above 50 EMA (avoid trading in downtrends)
            if (self.data['Close'].iloc[i] <= self.data['BB_Lower'].iloc[i] and
                self.data['RSI'].iloc[i] <= 40 and  # Raised from 35 to 40
                self.data['Close'].iloc[i] > self.data['EMA_50'].iloc[i]):  # 50 EMA filter
                return 1  # Buy signal
            
            # Enhanced short entry conditions:
            # 1. Price above upper band
            # 2. RSI overbought (kept at 65)
            # 3. Price below 50 EMA (for short trades in downtrends)
            elif (self.data['Close'].iloc[i] >= self.data['BB_Upper'].iloc[i] and
                  self.data['RSI'].iloc[i] >= 65 and
                  self.data['Close'].iloc[i] < self.data['EMA_50'].iloc[i]):  # 50 EMA filter for shorts
                return -1  # Sell signal
            
            return 0
        except (IndexError, KeyError):
            return 0
    
    def _get_bollinger_signal(self, i):
        """Legacy method - redirects to enhanced version"""
        return self._get_enhanced_bollinger_signal(i)
    
    def _track_positions_with_regime_switching(self):
        """Track positions with automatic closure on regime changes and enhanced exit conditions"""
        position = 0
        entry_bar = None
        entry_regime = None
        entry_price = None
        highest_price = None
        trailing_stop_price = None
        positions = []
        
        for i in range(len(self.data)):
            current_signal = self.data['Signal'].iloc[i]
            current_stable_regime = self.data['Stable_Regime'].iloc[i] if 'Stable_Regime' in self.data.columns else self.data['Market_Regime'].iloc[i]
            current_price = self.data['Close'].iloc[i]
            
            # Force close position if STABLE regime changes (reduces whipsaw)
            if position != 0 and entry_regime != current_stable_regime and current_stable_regime != 'none':
                # Log regime change exit
                if self.current_trade:
                    self.current_trade['exit_reason'] = 'regime_change'
                    self.current_trade['exit_price'] = current_price
                    self.current_trade['exit_bar'] = i
                    self._finalize_trade()
                
                position = 0
                entry_bar = None
                entry_regime = None
                entry_price = None
                highest_price = None
                trailing_stop_price = None
            
            # Entry signals
            elif current_signal == 1 and position == 0:  # Buy signal
                position = 1
                entry_bar = i
                entry_regime = current_stable_regime
                entry_price = current_price
                highest_price = current_price
                trailing_stop_price = None
                
                # Start new trade log
                self.current_trade = {
                    'entry_bar': i,
                    'entry_price': current_price,
                    'entry_regime': current_stable_regime,
                    'strategy': self.data['Active_Strategy'].iloc[i],
                    'position_type': 'long'
                }
                
            elif current_signal == -1 and position == 0:  # Sell signal
                position = -1
                entry_bar = i
                entry_regime = current_stable_regime
                entry_price = current_price
                highest_price = current_price  # For shorts, track lowest price
                trailing_stop_price = None
                
                # Start new trade log
                self.current_trade = {
                    'entry_bar': i,
                    'entry_price': current_price,
                    'entry_regime': current_stable_regime,
                    'strategy': self.data['Active_Strategy'].iloc[i],
                    'position_type': 'short'
                }
            
            # Update trailing stops and exit conditions
            elif position != 0:
                active_strategy = self.data['Active_Strategy'].iloc[i]
                
                # Update highest/lowest price for trailing stops
                if position == 1:  # Long position
                    if current_price > highest_price:
                        highest_price = current_price
                        
                        # Update trailing stop for EMA strategy after 2% profit (4% trailing stop)
                        if (active_strategy == 'ema' and 
                            (current_price - entry_price) / entry_price >= 0.02):
                            trailing_stop_price = current_price * (1 - 0.04)  # 4% trailing stop
                
                elif position == -1:  # Short position
                    if current_price < highest_price:  # For shorts, track lowest price
                        highest_price = current_price
                
                # Check exit conditions
                should_exit = False
                exit_reason = None
                
                if active_strategy == 'ema' and position == 1:
                    # EMA strategy exits
                    if self.data['EMA_5'].iloc[i] > self.data['EMA_21'].iloc[i]:
                        should_exit = True
                        exit_reason = 'ema_crossover'
                    elif trailing_stop_price and current_price <= trailing_stop_price:
                        should_exit = True
                        exit_reason = 'trailing_stop'
                    elif current_price <= entry_price * (1 - 0.03):  # Hard stop loss 3% (increased)
                        should_exit = True
                        exit_reason = 'stop_loss'
                    elif entry_bar is not None and (i - entry_bar) >= 25:  # Time-based exit (25 bars - increased)
                        should_exit = True
                        exit_reason = 'time_exit'
                
                elif active_strategy == 'bollinger':
                    # Bollinger exit conditions (enhanced)
                    if position == 1 and current_price >= self.data['BB_Upper'].iloc[i]:  # Exit at upper band instead of middle
                        should_exit = True
                        exit_reason = 'bb_upper_band'
                    elif position == -1 and current_price <= self.data['BB_Lower'].iloc[i]:
                        should_exit = True
                        exit_reason = 'bb_lower_band'
                    elif current_price <= entry_price * (1 - 0.03):  # Stop loss 3% (increased)
                        should_exit = True
                        exit_reason = 'stop_loss'
                    elif entry_bar is not None and (i - entry_bar) >= 15:  # Time-based exit (15 bars)
                        should_exit = True
                        exit_reason = 'time_exit'
                
                if should_exit:
                    # Log trade exit
                    if self.current_trade:
                        self.current_trade['exit_reason'] = exit_reason
                        self.current_trade['exit_price'] = current_price
                        self.current_trade['exit_bar'] = i
                        self._finalize_trade()
                    
                    position = 0
                    entry_bar = None
                    entry_regime = None
                    entry_price = None
                    highest_price = None
                    trailing_stop_price = None
            
            positions.append(position)
        
        self.data['Position'] = positions
    
    def backtest_strategy(self, initial_capital=10000, transaction_cost=0.001):
        """
        Backtest the Market-Adaptive Hybrid strategy
        
        Parameters:
        -----------
        initial_capital : float, default=10000
            Initial capital for backtesting
        transaction_cost : float, default=0.001
            Transaction cost as percentage (0.1%)
        """
        print("Running Market-Adaptive Hybrid backtest...")
        
        self.data['Returns'] = self.data['Close'].pct_change()
        
        # Apply stop losses and position sizing
        strategy_returns = []
        position = 0
        entry_price = None
        entry_regime = None
        daily_trades = {}
        
        for i in range(len(self.data)):
            current_position = self.data['Position'].iloc[i]
            current_price = self.data['Close'].iloc[i]
            current_date = self.data.index[i].date()
            daily_return = self.data['Returns'].iloc[i]
            
            # Track daily trades
            if current_date not in daily_trades:
                daily_trades[current_date] = 0
            
            # Track entry price and regime
            if current_position != 0 and position == 0:  # New position
                entry_price = current_price
                entry_regime = self.data['Market_Regime'].iloc[i]
                daily_trades[current_date] += 1
            elif current_position == 0 and position != 0:  # Exit position
                entry_price = None
                entry_regime = None
            
            # Check daily trade limit
            if daily_trades[current_date] > self.max_trades_per_day:
                current_position = 0
            
            # Check stop loss
            if position != 0 and entry_price is not None:
                if position == 1 and current_price <= entry_price * (1 - self.stop_loss_pct):
                    # Long stop loss hit
                    strategy_returns.append(-self.stop_loss_pct)
                    position = 0
                    entry_price = None
                    continue
                elif position == -1 and current_price >= entry_price * (1 + self.stop_loss_pct):
                    # Short stop loss hit
                    strategy_returns.append(-self.stop_loss_pct)
                    position = 0
                    entry_price = None
                    continue
            
            # Regular returns
            if i > 0:
                strategy_returns.append(self.data['Position'].iloc[i-1] * daily_return)
            else:
                strategy_returns.append(0)
            
            position = current_position
        
        self.data['Strategy_Returns'] = strategy_returns
        
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
        
        # Calculate performance metrics
        self._calculate_performance_metrics(initial_capital)
        
        # Calculate regime-specific performance
        self._calculate_regime_performance()
        
        print("Market-Adaptive Hybrid backtest completed!")
        return self.backtest_results
    
    def _calculate_performance_metrics(self, initial_capital):
        """Calculate comprehensive performance metrics"""
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
            
            avg_win = np.mean([ret for ret in trade_returns if ret > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([ret for ret in trade_returns if ret <= 0]) if (total_trades - winning_trades) > 0 else 0
            profit_factor = abs(avg_win * winning_trades / (avg_loss * (total_trades - winning_trades))) if avg_loss != 0 else float('inf')
        else:
            win_rate = 0
            total_trades = 0
            profit_factor = 0
        
        # Sharpe ratio (annualized for hourly data)
        strategy_std = self.data['Strategy_Returns'].std() * np.sqrt(365 * 24)
        strategy_mean = self.data['Strategy_Returns'].mean() * 365 * 24
        sharpe_ratio = strategy_mean / strategy_std if strategy_std != 0 else 0
        
        # Calculate profit factor from trade log
        if self.trade_log:
            gross_profit = sum(trade['pnl'] for trade in self.trade_log if trade['pnl'] > 0)
            gross_loss = abs(sum(trade['pnl'] for trade in self.trade_log if trade['pnl'] < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            profit_factor = 0
        
        self.backtest_results = {
            'Total Return (%)': total_return,
            'Market Return (%)': market_return,
            'Excess Return (%)': total_return - market_return,
            'Max Drawdown (%)': max_drawdown,
            'Win Rate (%)': win_rate,
            'Total Trades': total_trades,
            'Profit Factor': profit_factor,
            'Sharpe Ratio': sharpe_ratio,
            'Final Equity': self.data['Equity'].iloc[-1]
        }
    
    def _calculate_regime_performance(self):
        """Calculate performance metrics for each regime"""
        for regime in ['trending', 'range']:
            regime_data = self.data[self.data['Market_Regime'] == regime]
            if len(regime_data) > 0:
                regime_return = (regime_data['Strategy_Returns'] + 1).prod() - 1
                self.regime_stats[regime]['total_return'] = regime_return * 100
                
                # Count trades in this regime from trade log
                regime_trades = [trade for trade in self.trade_log if trade.get('entry_regime') == regime]
                if regime_trades:
                    winning_trades = sum(1 for trade in regime_trades if trade.get('pnl', 0) > 0)
                    total_trades = len(regime_trades)
                    
                    self.regime_stats[regime]['trades'] = total_trades
                    self.regime_stats[regime]['wins'] = winning_trades
    
    def print_performance_summary(self):
        """Print comprehensive performance summary"""
        if self.backtest_results is None:
            print("No backtest results available. Run backtest first.")
            return
        
        print("\n" + "="*60)
        print("MARKET-ADAPTIVE HYBRID STRATEGY RESULTS (ENHANCED)")
        print("="*60)
        
        print("\n📊 OVERALL PERFORMANCE:")
        for key, value in self.backtest_results.items():
            if isinstance(value, float):
                print(f"{key:<25}: {value:>10.2f}")
            else:
                print(f"{key:<25}: {value:>10}")
        
        print(f"\n🎯 REGIME-SPECIFIC PERFORMANCE:")
        for regime, stats in self.regime_stats.items():
            if stats['trades'] > 0:
                win_rate = (stats['wins'] / stats['trades']) * 100
                print(f"{regime.capitalize()} Markets:")
                print(f"  - Trades: {stats['trades']}")
                print(f"  - Win Rate: {win_rate:.1f}%")
                print(f"  - Total Return: {stats['total_return']:.2f}%")
        
        # Show recent trades
        if self.trade_log:
            print(f"\n📋 RECENT TRADES (Last 5):")
            recent_trades = self.trade_log[-5:] if len(self.trade_log) >= 5 else self.trade_log
            
            for i, trade in enumerate(recent_trades, 1):
                pnl_str = f"{trade['pnl_pct']:+.2f}%"
                status = "✅ WIN" if trade['pnl'] > 0 else "❌ LOSS"
                print(f"  {i}. {trade['strategy'].upper()} {trade['position_type']} - "
                      f"Entry: ${trade['entry_price']:.2f} → Exit: ${trade['exit_price']:.2f} "
                      f"({pnl_str}) {status} [{trade['exit_reason']}]")
            
            # Trade statistics by exit reason
            exit_reasons = {}
            for trade in self.trade_log:
                reason = trade.get('exit_reason', 'unknown')
                if reason not in exit_reasons:
                    exit_reasons[reason] = {'count': 0, 'wins': 0}
                exit_reasons[reason]['count'] += 1
                if trade['pnl'] > 0:
                    exit_reasons[reason]['wins'] += 1
            
            print(f"\n📊 EXIT REASON ANALYSIS:")
            for reason, stats in exit_reasons.items():
                win_rate = (stats['wins'] / stats['count']) * 100 if stats['count'] > 0 else 0
                print(f"  - {reason.replace('_', ' ').title()}: {stats['count']} trades ({win_rate:.1f}% win rate)")
            
            # Enhanced strategy performance breakdown
            strategy_stats = {}
            for trade in self.trade_log:
                strategy = trade.get('strategy', 'unknown')
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {'count': 0, 'wins': 0, 'total_pnl': 0}
                strategy_stats[strategy]['count'] += 1
                strategy_stats[strategy]['total_pnl'] += trade['pnl']
                if trade['pnl'] > 0:
                    strategy_stats[strategy]['wins'] += 1
            
            print(f"\n🎯 STRATEGY PERFORMANCE BREAKDOWN:")
            for strategy, stats in strategy_stats.items():
                if stats['count'] > 0:
                    win_rate = (stats['wins'] / stats['count']) * 100
                    avg_return = (stats['total_pnl'] / stats['count']) * 100
                    print(f"  - {strategy.upper()}: {stats['count']} trades, "
                          f"{win_rate:.1f}% win rate, {avg_return:+.2f}% avg return")
        
        print(f"\n📈 MARKET REGIME DISTRIBUTION:")
        regime_dist = self.data['Market_Regime'].value_counts()
        for regime, count in regime_dist.items():
            pct = (count / len(self.data)) * 100
            print(f"  - {regime.capitalize()}: {count} periods ({pct:.1f}%)")
        
        print("="*60)
    
    def _finalize_trade(self):
        """Finalize current trade and add to trade log"""
        if self.current_trade:
            # Calculate P&L
            entry_price = self.current_trade['entry_price']
            exit_price = self.current_trade['exit_price']
            position_type = self.current_trade['position_type']
            
            if position_type == 'long':
                pnl_pct = (exit_price - entry_price) / entry_price
            else:  # short
                pnl_pct = (entry_price - exit_price) / entry_price
            
            self.current_trade['pnl'] = pnl_pct
            self.current_trade['pnl_pct'] = pnl_pct * 100
            
            # Add to trade log
            self.trade_log.append(self.current_trade.copy())
            self.current_trade = None
    
    def calculate_position_size(self, current_equity):
        """Calculate position size with enhanced risk management"""
        max_equity = getattr(self, 'max_equity', current_equity)
        drawdown = (current_equity - max_equity) / max_equity if max_equity > 0 else 0
        
        # Less aggressive drawdown risk reduction
        if drawdown < -0.15:  # 15% drawdown
            risk_factor = 0.5
        elif drawdown < -0.08:  # 8% drawdown
            risk_factor = 0.75
        else:
            risk_factor = 1.0
        
        # Enhanced position sizing limits
        risk_amount = current_equity * self.risk_per_trade * risk_factor
        max_position_value = current_equity * 0.20  # Increased to 20% of equity
        
        return {'risk_amount': risk_amount, 'max_position_value': max_position_value, 'risk_factor': risk_factor}
    
    def plot_comprehensive_results(self, start_date=None, end_date=None, figsize=(18, 15)):
        """
        Plot comprehensive Market-Adaptive strategy results
        
        Parameters:
        -----------
        start_date : str, optional
            Start date for plotting (YYYY-MM-DD format)
        end_date : str, optional
            End date for plotting (YYYY-MM-DD format)
        figsize : tuple, default=(18, 15)
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
        
        fig, axes = plt.subplots(5, 1, figsize=figsize)
        fig.suptitle('Market-Adaptive Hybrid Strategy - Comprehensive Analysis', fontsize=16, fontweight='bold')
        
        # 1. Price with regime background and signals
        axes[0].plot(plot_data.index, plot_data['Close'], label='BTC Price', color='black', linewidth=1.5)
        
        # Color background based on regime
        regime_colors = {'trending': 'lightgreen', 'range': 'lightcoral', 'transition': 'lightyellow'}
        for regime, color in regime_colors.items():
            regime_data = plot_data[plot_data['Market_Regime'] == regime]
            if not regime_data.empty:
                for start_idx, end_idx in self._get_regime_periods(plot_data, regime):
                    axes[0].axvspan(plot_data.index[start_idx], plot_data.index[end_idx], 
                                   alpha=0.2, color=color)
        
        # Mark trading signals
        buy_signals = plot_data[plot_data['Position_Change'] == 1]
        exit_signals = plot_data[plot_data['Position_Change'] == -1]
        
        axes[0].scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', 
                       s=50, label='Entry Signal', zorder=5)
        axes[0].scatter(exit_signals.index, exit_signals['Close'], color='red', marker='v', 
                       s=50, label='Exit Signal', zorder=5)
        
        axes[0].set_title('Price with Market Regimes and Trading Signals')
        axes[0].set_ylabel('Price (USD)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. ADX with regime thresholds
        axes[1].plot(plot_data.index, plot_data['ADX'], label='ADX', color='purple', linewidth=2)
        axes[1].axhline(y=self.regime_detector.trend_threshold, color='green', linestyle='--', 
                       label=f'Trend Threshold ({self.regime_detector.trend_threshold})', alpha=0.7)
        axes[1].axhline(y=self.regime_detector.range_threshold, color='red', linestyle='--', 
                       label=f'Range Threshold ({self.regime_detector.range_threshold})', alpha=0.7)
        
        axes[1].set_title('ADX - Market Regime Detection')
        axes[1].set_ylabel('ADX Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Active strategy indicator
        strategy_mapping = {'bollinger': 1, 'ema': 2, 'transition': 0, 'none': 0}
        plot_data['Strategy_Numeric'] = plot_data['Active_Strategy'].map(strategy_mapping)
        
        axes[2].plot(plot_data.index, plot_data['Strategy_Numeric'], label='Active Strategy', 
                    color='orange', linewidth=2, drawstyle='steps-post')
        axes[2].set_title('Active Trading Strategy (0=None/Transition, 1=Bollinger, 2=EMA)')
        axes[2].set_ylabel('Strategy')
        axes[2].set_ylim(-0.5, 2.5)
        axes[2].set_yticks([0, 1, 2])
        axes[2].set_yticklabels(['None/Transition', 'Bollinger', 'EMA'])
        axes[2].grid(True, alpha=0.3)
        
        # 4. Position tracking
        axes[3].plot(plot_data.index, plot_data['Position'], label='Position', 
                    color='navy', linewidth=2, drawstyle='steps-post')
        axes[3].fill_between(plot_data.index, 0, plot_data['Position'], alpha=0.3, 
                           color='navy', step='post')
        axes[3].set_title('Trading Position (1=Long, -1=Short, 0=Neutral)')
        axes[3].set_ylabel('Position')
        axes[3].set_ylim(-1.5, 1.5)
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        # 5. Equity curve comparison
        axes[4].plot(plot_data.index, plot_data['Equity'], label='Hybrid Strategy', 
                    color='green', linewidth=2.5)
        axes[4].plot(plot_data.index, plot_data['Market_Equity'], label='Buy & Hold', 
                    color='blue', linewidth=2, alpha=0.7)
        axes[4].set_title('Equity Curve Comparison')
        axes[4].set_ylabel('Equity (USD)')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def _get_regime_periods(self, data, regime):
        """Get start and end indices for regime periods"""
        regime_mask = data['Market_Regime'] == regime
        periods = []
        start_idx = None
        
        for i, is_regime in enumerate(regime_mask):
            if is_regime and start_idx is None:
                start_idx = i
            elif not is_regime and start_idx is not None:
                periods.append((start_idx, i-1))
                start_idx = None
        
        # Handle case where regime continues to end
        if start_idx is not None:
            periods.append((start_idx, len(data)-1))
        
        return periods
    
    def run_complete_analysis(self, file_path, initial_capital=10000):
        """Run complete Market-Adaptive strategy analysis"""
        print("🚀 Starting Market-Adaptive Hybrid Strategy Analysis...")
        print("=" * 70)
        
        # Load data
        self.load_data(file_path)
        
        # Calculate all indicators
        self.calculate_all_indicators()
        
        # Generate hybrid signals
        self.generate_hybrid_signals()
        
        # Run backtest
        self.backtest_strategy(initial_capital=initial_capital)
        
        # Print results
        self.print_performance_summary()
        
        return self.data, self.backtest_results

# Example usage
if __name__ == "__main__":
    # Initialize Market-Adaptive strategy
    strategy = MarketAdaptiveStrategy()
    
    # Run complete analysis
    data_file = "BTCUSD_Hourly_Ask_2024.01.01_2024.12.31.csv"
    data, results = strategy.run_complete_analysis(data_file, initial_capital=10000)
    
    # Plot comprehensive results
    strategy.plot_comprehensive_results()
