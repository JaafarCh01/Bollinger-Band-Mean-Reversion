#!/usr/bin/env python3
"""
Market Regime Detection using ADX
Author: Trading Strategy Implementation
Date: 2025

This script implements ADX-based market regime detection to identify
trending vs range-bound market conditions for the hybrid strategy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class MarketRegimeDetector:
    def __init__(self, adx_period=14, trend_threshold=25, range_threshold=20):
        """
        Initialize Market Regime Detector
        
        Parameters:
        -----------
        adx_period : int, default=14
            Period for ADX calculation
        trend_threshold : float, default=25
            ADX threshold for trending market (ADX > threshold)
        range_threshold : float, default=20
            ADX threshold for range-bound market (ADX < threshold)
        """
        self.adx_period = adx_period
        self.trend_threshold = trend_threshold
        self.range_threshold = range_threshold
        
        self.data = None
    
    def calculate_true_range(self, data):
        """Calculate True Range (TR)"""
        data['Previous_Close'] = data['Close'].shift(1)
        
        # True Range is the maximum of:
        # 1. Current High - Current Low
        # 2. Current High - Previous Close (absolute)
        # 3. Current Low - Previous Close (absolute)
        data['TR'] = np.maximum(
            data['High'] - data['Low'],
            np.maximum(
                abs(data['High'] - data['Previous_Close']),
                abs(data['Low'] - data['Previous_Close'])
            )
        )
        
        return data
    
    def calculate_directional_movement(self, data):
        """Calculate Directional Movement (+DM and -DM)"""
        # Plus Directional Movement (+DM)
        data['Plus_DM'] = np.where(
            (data['High'] - data['High'].shift(1)) > (data['Low'].shift(1) - data['Low']),
            np.maximum(data['High'] - data['High'].shift(1), 0),
            0
        )
        
        # Minus Directional Movement (-DM)
        data['Minus_DM'] = np.where(
            (data['Low'].shift(1) - data['Low']) > (data['High'] - data['High'].shift(1)),
            np.maximum(data['Low'].shift(1) - data['Low'], 0),
            0
        )
        
        return data
    
    def calculate_adx(self, data):
        """
        Calculate ADX (Average Directional Index)
        
        ADX measures the strength of a trend, regardless of direction:
        - ADX > 25: Strong trend
        - ADX 20-25: Weak trend/transition
        - ADX < 20: Range-bound/sideways market
        """
        # Calculate True Range and Directional Movements
        data = self.calculate_true_range(data)
        data = self.calculate_directional_movement(data)
        
        # Smooth TR, +DM, and -DM using Wilder's smoothing (similar to EMA)
        alpha = 1.0 / self.adx_period
        
        data['ATR'] = data['TR'].ewm(alpha=alpha, adjust=False).mean()
        data['Plus_DI_Raw'] = data['Plus_DM'].ewm(alpha=alpha, adjust=False).mean()
        data['Minus_DI_Raw'] = data['Minus_DM'].ewm(alpha=alpha, adjust=False).mean()
        
        # Calculate Directional Indicators (+DI and -DI)
        data['Plus_DI'] = 100 * (data['Plus_DI_Raw'] / data['ATR'])
        data['Minus_DI'] = 100 * (data['Minus_DI_Raw'] / data['ATR'])
        
        # Calculate Directional Index (DX)
        data['DX'] = 100 * abs(data['Plus_DI'] - data['Minus_DI']) / (data['Plus_DI'] + data['Minus_DI'])
        
        # Calculate ADX (smoothed DX)
        data['ADX'] = data['DX'].ewm(alpha=alpha, adjust=False).mean()
        
        return data
    
    def detect_regime(self, data):
        """
        Detect market regime based on ADX values
        
        Returns:
        --------
        regime : str
            'trending', 'range', or 'transition'
        """
        # Add regime classification
        conditions = [
            data['ADX'] > self.trend_threshold,
            data['ADX'] < self.range_threshold
        ]
        
        choices = ['trending', 'range']
        
        data['Market_Regime'] = np.select(conditions, choices, default='transition')
        
        return data
    
    def analyze_regime_distribution(self, data):
        """Analyze the distribution of market regimes"""
        regime_counts = data['Market_Regime'].value_counts()
        regime_percentages = (regime_counts / len(data) * 100).round(2)
        
        print("\n" + "="*40)
        print("MARKET REGIME ANALYSIS")
        print("="*40)
        print(f"Total periods analyzed: {len(data)}")
        print("\nRegime Distribution:")
        
        for regime in ['trending', 'range', 'transition']:
            if regime in regime_percentages:
                print(f"{regime.capitalize():<12}: {regime_counts[regime]:>6} periods ({regime_percentages[regime]:>5.1f}%)")
        
        print("="*40)
        
        return regime_counts, regime_percentages
    
    def plot_adx_analysis(self, data, start_date=None, end_date=None, figsize=(15, 10)):
        """
        Plot ADX analysis with regime identification
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with ADX calculations
        start_date : str, optional
            Start date for plotting
        end_date : str, optional
            End date for plotting
        figsize : tuple, default=(15, 10)
            Figure size
        """
        # Filter data for plotting if dates specified
        plot_data = data.copy()
        if start_date:
            plot_data = plot_data[plot_data.index >= start_date]
        if end_date:
            plot_data = plot_data[plot_data.index <= end_date]
        
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        fig.suptitle('ADX Market Regime Analysis', fontsize=16, fontweight='bold')
        
        # 1. Price with regime background
        axes[0].plot(plot_data.index, plot_data['Close'], label='BTC Price', color='black', linewidth=1)
        
        # Color background based on regime
        for regime, color in [('trending', 'lightgreen'), ('range', 'lightcoral'), ('transition', 'lightyellow')]:
            regime_data = plot_data[plot_data['Market_Regime'] == regime]
            if not regime_data.empty:
                axes[0].scatter(regime_data.index, regime_data['Close'], 
                              c=color, alpha=0.3, s=1, label=f'{regime.capitalize()} Market')
        
        axes[0].set_title('BTC Price with Market Regime Classification')
        axes[0].set_ylabel('Price (USD)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. ADX with thresholds
        axes[1].plot(plot_data.index, plot_data['ADX'], label='ADX', color='purple', linewidth=2)
        axes[1].axhline(y=self.trend_threshold, color='green', linestyle='--', 
                       label=f'Trend Threshold ({self.trend_threshold})', alpha=0.7)
        axes[1].axhline(y=self.range_threshold, color='red', linestyle='--', 
                       label=f'Range Threshold ({self.range_threshold})', alpha=0.7)
        
        # Fill regions
        axes[1].fill_between(plot_data.index, self.trend_threshold, 100, alpha=0.1, color='green', label='Trending Zone')
        axes[1].fill_between(plot_data.index, 0, self.range_threshold, alpha=0.1, color='red', label='Range Zone')
        axes[1].fill_between(plot_data.index, self.range_threshold, self.trend_threshold, 
                           alpha=0.1, color='yellow', label='Transition Zone')
        
        axes[1].set_title('ADX (Average Directional Index)')
        axes[1].set_ylabel('ADX Value')
        axes[1].set_ylim(0, max(plot_data['ADX'].max() * 1.1, 50))
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Directional Indicators
        axes[2].plot(plot_data.index, plot_data['Plus_DI'], label='+DI', color='green', alpha=0.8)
        axes[2].plot(plot_data.index, plot_data['Minus_DI'], label='-DI', color='red', alpha=0.8)
        axes[2].set_title('Directional Indicators (+DI and -DI)')
        axes[2].set_ylabel('DI Value')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_regime_analysis(self, data):
        """
        Run complete market regime analysis
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data with OHLCV columns
            
        Returns:
        --------
        data : pd.DataFrame
            Data with ADX and regime classifications
        """
        print("Running Market Regime Analysis...")
        print("-" * 40)
        
        # Calculate ADX
        data = self.calculate_adx(data)
        
        # Detect regimes
        data = self.detect_regime(data)
        
        # Analyze distribution
        regime_counts, regime_percentages = self.analyze_regime_distribution(data)
        
        print(f"\nADX Parameters:")
        print(f"- Period: {self.adx_period}")
        print(f"- Trend threshold: {self.trend_threshold}")
        print(f"- Range threshold: {self.range_threshold}")
        
        self.data = data
        return data

# Example usage
if __name__ == "__main__":
    # Load sample data
    import pandas as pd
    
    # Initialize regime detector
    detector = MarketRegimeDetector()
    
    # Load data (you would replace this with your actual data loading)
    print("Loading sample data...")
    data = pd.read_csv("BTCUSD_Hourly_Ask_2024.01.01_2024.12.31.csv")
    data['Time'] = pd.to_datetime(data['Time (EET)'])
    data.set_index('Time', inplace=True)
    
    # Drop the original time column and clean column names
    data.drop('Time (EET)', axis=1, inplace=True)
    
    # Clean column names (strip any whitespace)
    data.columns = data.columns.str.strip()
    
    # Ensure we have the expected columns
    expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if list(data.columns) != expected_columns:
        data.columns = expected_columns
    
    # Run regime analysis
    data_with_regime = detector.run_regime_analysis(data)
    
    # Plot results
    detector.plot_adx_analysis(data_with_regime)
