import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from matplotlib.ticker import FuncFormatter
from stock_analyser.utils.constants import FONT_FAMILY

# Set the style for all plots
plt.style.use('seaborn-v0_8-darkgrid')

def calculate_williams_r(high, low, close, period=14):
    """Calculate Williams %R indicator"""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    return williams_r

def calculate_ultimate_oscillator(high, low, close, period1=7, period2=14, period3=28):
    """Calculate Ultimate Oscillator"""
    true_range = pd.DataFrame({
        'tr1': high - low,
        'tr2': (high - close.shift()).abs(),
        'tr3': (low - close.shift()).abs()
    }).max(axis=1)
    
    buying_pressure = close - pd.concat([low, close.shift()], axis=1).min(axis=1)
    
    avg_bp1 = buying_pressure.rolling(period1).sum()
    avg_tr1 = true_range.rolling(period1).sum()
    
    avg_bp2 = buying_pressure.rolling(period2).sum()
    avg_tr2 = true_range.rolling(period2).sum()
    
    avg_bp3 = buying_pressure.rolling(period3).sum()
    avg_tr3 = true_range.rolling(period3).sum()
    
    uo = 100 * ((4 * avg_bp1 / avg_tr1) + (2 * avg_bp2 / avg_tr2) + (avg_bp3 / avg_tr3)) / (4 + 2 + 1)
    return uo

def calculate_schaff_trend_cycle(close, period=23, k_period=10, d_period=3):
    """Calculate Schaff Trend Cycle (STC)"""
    # MACD
    ema_fast = close.ewm(span=12).mean()
    ema_slow = close.ewm(span=26).mean()
    macd = ema_fast - ema_slow
    
    # Stochastic of MACD
    macd_min = macd.rolling(period).min()
    macd_max = macd.rolling(period).max()
    
    stoch_k = 100 * (macd - macd_min) / (macd_max - macd_min)
    stoch_k = stoch_k.rolling(k_period).mean()
    
    # Stochastic of Stochastic
    stoch_k_min = stoch_k.rolling(period).min()
    stoch_k_max = stoch_k.rolling(period).max()
    
    stc = 100 * (stoch_k - stoch_k_min) / (stoch_k_max - stoch_k_min)
    stc = stc.rolling(d_period).mean()
    
    return stc

def calculate_rsi(close, period=14):
    """Calculate RSI"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def money_formatter(x, pos):
    """Format y-axis labels as currency"""
    return f'${x:,.2f}'

def get_timeframe_data(ticker, timeframes):
    """
    Get data for multiple timeframes
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    timeframes : dict
        Dictionary with timeframe names and periods
        
    Returns:
    --------
    dict
        Dictionary containing dataframes for each timeframe
    """
    data = {}
    stock = yf.Ticker(ticker)
    
    for tf_name, period in timeframes.items():
        try:
            df = stock.history(period=period)
            if not df.empty:
                data[tf_name] = df
            else:
                print(f"⚠️ No data available for {ticker} on {tf_name} timeframe")
        except Exception as e:
            print(f"❌ Error fetching {tf_name} data for {ticker}: {str(e)}")
    
    return data

def calculate_momentum_indicators(df):
    """
    Calculate momentum indicators for a given dataframe
    
    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV dataframe
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with momentum indicators added
    """
    result = df.copy()
    
    # RSI
    result['RSI'] = calculate_rsi(df['Close'])
    
    # Williams %R
    result['Williams_R'] = calculate_williams_r(df['High'], df['Low'], df['Close'])
    
    # Ultimate Oscillator
    result['Ultimate_Osc'] = calculate_ultimate_oscillator(df['High'], df['Low'], df['Close'])
    
    # Schaff Trend Cycle
    result['STC'] = calculate_schaff_trend_cycle(df['Close'])
    
    # Stochastic
    low_min = df['Low'].rolling(window=14).min()
    high_max = df['High'].rolling(window=14).max()
    result['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    result['Stoch_D'] = result['Stoch_K'].rolling(window=3).mean()
    
    return result

def plot_multi_timeframe_momentum(ticker, output_dir='plots', timestamp=None):
    """
    Create a multi-timeframe momentum dashboard
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    output_dir : str
        Directory to save the plot
    timestamp : str
        Timestamp for filename
        
    Returns:
    --------
    str
        Path to the saved plot file
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define timeframes
    timeframes = {
        'Daily': '6mo',      # 6 months of daily data
        'Weekly': '2y',      # 2 years of weekly data  
        'Monthly': '5y'      # 5 years of monthly data
    }
    
    print(f"Fetching multi-timeframe data for {ticker}...")
    
    # Get data for all timeframes
    tf_data = get_timeframe_data(ticker, timeframes)
    
    if not tf_data:
        print(f"❌ No data available for {ticker}")
        return None
    
    # Calculate indicators for each timeframe
    tf_indicators = {}
    for tf_name, df in tf_data.items():
        print(f"Calculating indicators for {tf_name} timeframe...")
        tf_indicators[tf_name] = calculate_momentum_indicators(df)
    
    # Create the plot
    fig = plt.figure(figsize=(18, 16), dpi=300)
    
    # Define grid layout: 5 rows x 3 columns
    gs = fig.add_gridspec(5, 3, height_ratios=[2, 1, 1, 1, 1], hspace=0.3, wspace=0.3)
    
    # Colors for timeframes
    colors = {'Daily': '#1f77b4', 'Weekly': '#ff7f0e', 'Monthly': '#2ca02c'}
    
    # Row 1: Price charts for each timeframe
    price_axes = []
    for i, (tf_name, df) in enumerate(tf_indicators.items()):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(df.index, df['Close'], color=colors[tf_name], linewidth=2, label=f'{tf_name} Close')
        ax.set_title(f'{ticker} - {tf_name} Price', fontsize=14, fontweight='bold', fontfamily=FONT_FAMILY)
        ax.yaxis.set_major_formatter(FuncFormatter(money_formatter))
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis based on timeframe
        if tf_name == 'Daily':
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        elif tf_name == 'Weekly':
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        else:  # Monthly
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        price_axes.append(ax)
    
    # Row 2: RSI Comparison
    ax_rsi = fig.add_subplot(gs[1, :])
    for tf_name, df in tf_indicators.items():
        if 'RSI' in df.columns and not df['RSI'].isnull().all():
            ax_rsi.plot(df.index, df['RSI'], color=colors[tf_name], linewidth=2, 
                       label=f'{tf_name} RSI', alpha=0.8)
    
    ax_rsi.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
    ax_rsi.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
    ax_rsi.fill_between(ax_rsi.get_xlim(), 70, 100, alpha=0.1, color='red')
    ax_rsi.fill_between(ax_rsi.get_xlim(), 0, 30, alpha=0.1, color='green')
    ax_rsi.set_title('RSI Comparison Across Timeframes', fontsize=14, fontweight='bold', fontfamily=FONT_FAMILY)
    ax_rsi.set_ylabel('RSI', fontfamily=FONT_FAMILY)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.grid(True, alpha=0.3)
    ax_rsi.legend(loc='upper right')
    
    # Row 3: Williams %R Comparison
    ax_wr = fig.add_subplot(gs[2, :])
    for tf_name, df in tf_indicators.items():
        if 'Williams_R' in df.columns and not df['Williams_R'].isnull().all():
            ax_wr.plot(df.index, df['Williams_R'], color=colors[tf_name], linewidth=2, 
                      label=f'{tf_name} Williams %R', alpha=0.8)
    
    ax_wr.axhline(y=-20, color='r', linestyle='--', alpha=0.5, label='Overbought (-20)')
    ax_wr.axhline(y=-80, color='g', linestyle='--', alpha=0.5, label='Oversold (-80)')
    ax_wr.fill_between(ax_wr.get_xlim(), -20, 0, alpha=0.1, color='red')
    ax_wr.fill_between(ax_wr.get_xlim(), -100, -80, alpha=0.1, color='green')
    ax_wr.set_title('Williams %R Comparison Across Timeframes', fontsize=14, fontweight='bold', fontfamily=FONT_FAMILY)
    ax_wr.set_ylabel('Williams %R', fontfamily=FONT_FAMILY)
    ax_wr.set_ylim(-100, 0)
    ax_wr.grid(True, alpha=0.3)
    ax_wr.legend(loc='upper right')
    
    # Row 4: Ultimate Oscillator Comparison
    ax_uo = fig.add_subplot(gs[3, :])
    for tf_name, df in tf_indicators.items():
        if 'Ultimate_Osc' in df.columns and not df['Ultimate_Osc'].isnull().all():
            ax_uo.plot(df.index, df['Ultimate_Osc'], color=colors[tf_name], linewidth=2, 
                      label=f'{tf_name} Ultimate Oscillator', alpha=0.8)
    
    ax_uo.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
    ax_uo.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
    ax_uo.fill_between(ax_uo.get_xlim(), 70, 100, alpha=0.1, color='red')
    ax_uo.fill_between(ax_uo.get_xlim(), 0, 30, alpha=0.1, color='green')
    ax_uo.set_title('Ultimate Oscillator Comparison Across Timeframes', fontsize=14, fontweight='bold', fontfamily=FONT_FAMILY)
    ax_uo.set_ylabel('Ultimate Oscillator', fontfamily=FONT_FAMILY)
    ax_uo.set_ylim(0, 100)
    ax_uo.grid(True, alpha=0.3)
    ax_uo.legend(loc='upper right')
    
    # Row 5: Schaff Trend Cycle Comparison
    ax_stc = fig.add_subplot(gs[4, :])
    for tf_name, df in tf_indicators.items():
        if 'STC' in df.columns and not df['STC'].isnull().all():
            ax_stc.plot(df.index, df['STC'], color=colors[tf_name], linewidth=2, 
                       label=f'{tf_name} Schaff Trend Cycle', alpha=0.8)
    
    ax_stc.axhline(y=75, color='r', linestyle='--', alpha=0.5, label='Overbought (75)')
    ax_stc.axhline(y=25, color='g', linestyle='--', alpha=0.5, label='Oversold (25)')
    ax_stc.fill_between(ax_stc.get_xlim(), 75, 100, alpha=0.1, color='red')
    ax_stc.fill_between(ax_stc.get_xlim(), 0, 25, alpha=0.1, color='green')
    ax_stc.set_title('Schaff Trend Cycle Comparison Across Timeframes', fontsize=14, fontweight='bold', fontfamily=FONT_FAMILY)
    ax_stc.set_ylabel('STC', fontfamily=FONT_FAMILY)
    ax_stc.set_ylim(0, 100)
    ax_stc.grid(True, alpha=0.3)
    ax_stc.legend(loc='upper right')
    
    # Add overall title
    fig.suptitle(f'Multi-Timeframe Momentum Dashboard: {ticker}', 
                fontsize=18, fontweight='bold', y=0.98, fontfamily=FONT_FAMILY)
    
    # Add summary text box
    latest_signals = analyze_momentum_signals(tf_indicators)
    textstr = format_signal_summary(latest_signals)
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8)
    fig.text(0.02, 0.02, textstr, fontsize=10, bbox=props, fontfamily=FONT_FAMILY)
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    filename = f'{ticker}_multi_timeframe_momentum_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    
    # Save the plot
    plt.savefig(filepath, bbox_inches='tight', dpi=300, transparent=False)
    plt.close(fig)
    
    print(f"📊 Multi-timeframe momentum dashboard saved to {filepath}")
    return filepath

def analyze_momentum_signals(tf_indicators):
    """
    Analyze current momentum signals across timeframes
    
    Parameters:
    -----------
    tf_indicators : dict
        Dictionary containing indicator data for each timeframe
        
    Returns:
    --------
    dict
        Signal analysis for each timeframe and indicator
    """
    signals = {}
    
    for tf_name, df in tf_indicators.items():
        if df.empty:
            continue
            
        signals[tf_name] = {}
        
        # Get latest values
        latest = df.iloc[-1]
        
        # RSI signals
        if 'RSI' in df.columns and not pd.isna(latest['RSI']):
            rsi = latest['RSI']
            if rsi > 70:
                signals[tf_name]['RSI'] = 'Overbought'
            elif rsi < 30:
                signals[tf_name]['RSI'] = 'Oversold'
            else:
                signals[tf_name]['RSI'] = 'Neutral'
        
        # Williams %R signals  
        if 'Williams_R' in df.columns and not pd.isna(latest['Williams_R']):
            wr = latest['Williams_R']
            if wr > -20:
                signals[tf_name]['Williams_R'] = 'Overbought'
            elif wr < -80:
                signals[tf_name]['Williams_R'] = 'Oversold'
            else:
                signals[tf_name]['Williams_R'] = 'Neutral'
        
        # Ultimate Oscillator signals
        if 'Ultimate_Osc' in df.columns and not pd.isna(latest['Ultimate_Osc']):
            uo = latest['Ultimate_Osc']
            if uo > 70:
                signals[tf_name]['Ultimate_Osc'] = 'Overbought'
            elif uo < 30:
                signals[tf_name]['Ultimate_Osc'] = 'Oversold'
            else:
                signals[tf_name]['Ultimate_Osc'] = 'Neutral'
        
        # STC signals
        if 'STC' in df.columns and not pd.isna(latest['STC']):
            stc = latest['STC']
            if stc > 75:
                signals[tf_name]['STC'] = 'Overbought'
            elif stc < 25:
                signals[tf_name]['STC'] = 'Oversold'
            else:
                signals[tf_name]['STC'] = 'Neutral'
    
    return signals

def format_signal_summary(signals):
    """Format signal analysis into readable text"""
    summary_lines = ["MOMENTUM SIGNAL SUMMARY:"]
    
    for tf_name, tf_signals in signals.items():
        if tf_signals:
            summary_lines.append(f"\n{tf_name.upper()}:")
            for indicator, signal in tf_signals.items():
                summary_lines.append(f"  {indicator}: {signal}")
    
    return "\n".join(summary_lines)

def get_momentum_consensus(ticker, output_dir='plots', timestamp=None):
    """
    Generate momentum consensus across timeframes
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    output_dir : str
        Directory to save the plot
    timestamp : str
        Timestamp for filename
        
    Returns:
    --------
    dict
        Consensus analysis results
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get timeframe data
    timeframes = {
        'Daily': '6mo',
        'Weekly': '2y', 
        'Monthly': '5y'
    }
    
    tf_data = get_timeframe_data(ticker, timeframes)
    if not tf_data:
        return None
    
    # Calculate indicators
    tf_indicators = {}
    for tf_name, df in tf_data.items():
        tf_indicators[tf_name] = calculate_momentum_indicators(df)
    
    # Analyze signals
    signals = analyze_momentum_signals(tf_indicators)
    
    # Calculate consensus
    consensus = {}
    indicators = ['RSI', 'Williams_R', 'Ultimate_Osc', 'STC']
    
    for indicator in indicators:
        bullish = 0
        bearish = 0
        neutral = 0
        
        for tf_name in timeframes.keys():
            if tf_name in signals and indicator in signals[tf_name]:
                signal = signals[tf_name][indicator]
                if signal == 'Oversold':
                    bullish += 1
                elif signal == 'Overbought':
                    bearish += 1
                else:
                    neutral += 1
        
        total_signals = bullish + bearish + neutral
        if total_signals > 0:
            consensus[indicator] = {
                'bullish_pct': (bullish / total_signals) * 100,
                'bearish_pct': (bearish / total_signals) * 100,
                'neutral_pct': (neutral / total_signals) * 100,
                'consensus': 'Bullish' if bullish > bearish else 'Bearish' if bearish > bullish else 'Neutral'
            }
    
    return {
        'signals': signals,
        'consensus': consensus,
        'plot_path': None  # Will be set when plot is generated
    }

if __name__ == "__main__":
    # Example usage
    plot_path = plot_multi_timeframe_momentum('AAPL', 'test_plots')
    if plot_path:
        print(f"Multi-timeframe momentum dashboard saved to: {plot_path}")
        
    # Get consensus analysis
    consensus = get_momentum_consensus('AAPL', 'test_plots')
    if consensus:
        print("\nMomentum Consensus Analysis:")
        for indicator, data in consensus['consensus'].items():
            print(f"{indicator}: {data['consensus']} ({data['bullish_pct']:.1f}% bullish, {data['bearish_pct']:.1f}% bearish)")