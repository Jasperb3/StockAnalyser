import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from mplfinance.original_flavor import candlestick_ohlc

# Set the style for all plots
plt.style.use('seaborn-v0_8-darkgrid')

def get_stock_price(symbol, start_date='2021-02-01'):
    """Get stock price data using yfinance"""
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date)
    df['Date'] = pd.to_datetime(df.index)
    df['Date'] = df['Date'].apply(mpl_dates.date2num)
    df = df.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]
    return df

def is_support(df, i):
    """Check if the candle at position i is a support level"""
    cond1 = df['Low'].iloc[i] < df['Low'].iloc[i-1] 
    cond2 = df['Low'].iloc[i] < df['Low'].iloc[i+1] 
    cond3 = df['Low'].iloc[i+1] < df['Low'].iloc[i+2] 
    cond4 = df['Low'].iloc[i-1] < df['Low'].iloc[i-2]
    return (cond1 and cond2 and cond3 and cond4)

def is_resistance(df, i):
    """Check if the candle at position i is a resistance level"""
    cond1 = df['High'].iloc[i] > df['High'].iloc[i-1] 
    cond2 = df['High'].iloc[i] > df['High'].iloc[i+1] 
    cond3 = df['High'].iloc[i+1] > df['High'].iloc[i+2] 
    cond4 = df['High'].iloc[i-1] > df['High'].iloc[i-2]
    return (cond1 and cond2 and cond3 and cond4)

def is_far_from_level(value, levels, df):
    """Check if the value is far enough from existing levels"""
    ave = np.mean(df['High'] - df['Low'])
    return np.sum([abs(value - level) < ave for _, level in levels]) == 0

def money_formatter(x, pos):
    """Format y-axis labels as currency"""
    return f'${x:,.2f}'

def plot_breakout_levels(df, levels, symbol, method_name, output_dir='plots', timestamp=None):
    """Plot candlestick chart with support/resistance levels"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    
    # Plot candlestick chart
    candlestick_ohlc(ax, df.values, width=0.6, colorup='#2ca02c', colordown='#d62728', alpha=0.8)
    
    # Format axes
    ax.xaxis.set_major_formatter(mpl_dates.DateFormatter('%Y-%m-%d'))
    
    # Reduce the number of x-axis labels to approximately 25% of the original
    ax.xaxis.set_major_locator(mpl_dates.WeekdayLocator(interval=8))  # Show approximately every 8 weeks
    
    ax.yaxis.set_major_formatter(FuncFormatter(money_formatter))
    
    # Plot horizontal lines for support/resistance levels
    support_levels = [(i, level) for i, level in levels if level <= df['Close'].iloc[-1]]
    resistance_levels = [(i, level) for i, level in levels if level > df['Close'].iloc[-1]]
    
    # Plot support levels in green
    for level in support_levels:
        idx = level[0]
        price_level = level[1]
        date_val = df['Date'].iloc[idx] if idx < len(df) else df['Date'].iloc[0]
        ax.axhline(y=price_level, xmin=date_val/df['Date'].iloc[-1], xmax=1, 
                  color='#2ca02c', linestyle='--', linewidth=1.5, alpha=0.8)
        # Add price label
        ax.text(df['Date'].iloc[-1] + 1, price_level, f'${price_level:.2f}', 
                va='center', ha='left', fontsize=9, color='#2ca02c', fontweight='bold')
    
    # Plot resistance levels in red
    for level in resistance_levels:
        idx = level[0]
        price_level = level[1]
        date_val = df['Date'].iloc[idx] if idx < len(df) else df['Date'].iloc[0]
        ax.axhline(y=price_level, xmin=date_val/df['Date'].iloc[-1], xmax=1, 
                  color='#d62728', linestyle='--', linewidth=1.5, alpha=0.8)
        # Add price label
        ax.text(df['Date'].iloc[-1] + 1, price_level, f'${price_level:.2f}', 
                va='center', ha='left', fontsize=9, color='#d62728', fontweight='bold')
    
    # Add title and labels
    ax.set_title(f'{symbol} - Support & Resistance Levels ({method_name})', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12, labelpad=10)
    ax.set_ylabel('Price ($)', fontsize=12, labelpad=10)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add legend
    
    legend_elements = [
        Line2D([0], [0], color='#2ca02c', linestyle='--', lw=2, label='Support Level'),
        Line2D([0], [0], color='#d62728', linestyle='--', lw=2, label='Resistance Level'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ca02c', markersize=10, label='Bullish Candle'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#d62728', markersize=10, label='Bearish Candle')
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True, framealpha=0.9)
    
    # Add info box with level counts - MOVED DOWN to avoid overlap with legend
    textstr = f'Support Levels: {len(support_levels)}\nResistance Levels: {len(resistance_levels)}'
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
    ax.text(0.05, 0.80, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save figure
    filename = f'{symbol}_breakout_{method_name}_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved {method_name} breakout plot to {filepath}")
    
    return filepath

def detect_level_method_1(df):
    """Detect support/resistance levels using fractal candlestick pattern"""
    levels = []
    for i in range(2, len(df)-2):
        if is_support(df, i):
            l = df['Low'].iloc[i]
            if is_far_from_level(l, levels, df):
                levels.append((i, l))
        elif is_resistance(df, i):
            l = df['High'].iloc[i]
            if is_far_from_level(l, levels, df):
                levels.append((i, l))
    return levels

def detect_level_method_2(df):
    """Detect support/resistance levels using window shifting method"""
    levels = []
    max_list = []
    min_list = []
    for i in range(5, len(df)-5):
        # Check for resistance levels
        high_range = df['High'].iloc[i-5:i+4]
        current_max = high_range.max()

        if current_max not in max_list:
            max_list = []
        max_list.append(current_max)
        if len(max_list) == 5 and is_far_from_level(current_max, levels, df):
            max_idx = high_range.idxmax()
            idx_in_df = df.index.get_loc(max_idx)
            levels.append((idx_in_df, current_max))
        
        # Check for support levels
        low_range = df['Low'].iloc[i-5:i+5]
        current_min = low_range.min()
        if current_min not in min_list:
            min_list = []
        min_list.append(current_min)
        if len(min_list) == 5 and is_far_from_level(current_min, levels, df):
            min_idx = low_range.idxmin()
            idx_in_df = df.index.get_loc(min_idx)
            levels.append((idx_in_df, current_min))
    
    return levels

def has_breakout(levels, previous, last):
    """Check if there's a breakout through any of the levels"""
    for _, level in levels:
        cond1 = (previous['Open'] < level)  # to make sure previous candle is below the level
        cond2 = (last['Open'] > level) and (last['Low'] > level)
        if cond1 and cond2:
            return True
    return False

def get_sp500_stocks():
    """Get the list of S&P 500 stocks"""
    try:
        payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        return payload[0]['Symbol'].values.tolist()
    except Exception as e:
        print(f"Error fetching S&P 500 stocks: {str(e)}")
        return []

def screen_for_breakouts(stock_list, start_date='2021-02-01', max_stocks=None):
    """Screen stocks for breakout patterns"""
    if max_stocks and max_stocks < len(stock_list):
        stock_list = stock_list[:max_stocks]
    
    screened_list_1 = []  # Method 1 results
    screened_list_2 = []  # Method 2 results
    
    total = len(stock_list)
    print(f"Screening {total} stocks for breakout patterns...")
    
    for i, symbol in enumerate(stock_list):
        try:
            print(f"Processing {symbol} ({i+1}/{total})...", end='\r')
            df = get_stock_price(symbol, start_date)
            
            # Method 1: Fractal candlestick pattern
            levels_1 = detect_level_method_1(df)
            if levels_1 and len(levels_1) >= 5 and has_breakout(levels_1[-5:], df.iloc[-2], df.iloc[-1]):
                screened_list_1.append(symbol)
            
            # Method 2: Window shifting method
            levels_2 = detect_level_method_2(df)
            if levels_2 and len(levels_2) >= 5 and has_breakout(levels_2[-5:], df.iloc[-2], df.iloc[-1]):
                screened_list_2.append(symbol)
                
        except Exception as e:
            print(f"\nError processing {symbol}: {str(e)}")
    
    print("\nScreening complete.")
    return screened_list_1, screened_list_2

def analyze_single_stock(symbol, start_date='2021-02-01', output_dir='plots', timestamp=None):
    """Analyze a single stock for breakout patterns using both methods"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Analyzing {symbol} for breakout patterns...")
    
    # Get stock data
    df = get_stock_price(symbol, start_date)
    
    # Method 1: Fractal candlestick pattern
    print(f"Detecting support/resistance levels using Method 1 (Fractal Pattern)...")
    levels_1 = detect_level_method_1(df)
    
    # Method 2: Window shifting method
    print(f"Detecting support/resistance levels using Method 2 (Window Shifting)...")
    levels_2 = detect_level_method_2(df)
    
    # Plot results
    print(f"Plotting breakout levels for {symbol}...")
    plot_1 = plot_breakout_levels(df, levels_1, symbol, "Fractal_Pattern", output_dir, timestamp)
    plot_2 = plot_breakout_levels(df, levels_2, symbol, "Window_Shifting", output_dir, timestamp)
    
    # Check for breakouts
    has_breakout_1 = has_breakout(levels_1[-5:] if len(levels_1) >= 5 else levels_1, df.iloc[-2], df.iloc[-1])
    has_breakout_2 = has_breakout(levels_2[-5:] if len(levels_2) >= 5 else levels_2, df.iloc[-2], df.iloc[-1])
    
    results = {
        'symbol': symbol,
        'method1': {
            'levels': len(levels_1),
            'has_breakout': has_breakout_1,
            'plot_path': plot_1
        },
        'method2': {
            'levels': len(levels_2),
            'has_breakout': has_breakout_2,
            'plot_path': plot_2
        }
    }
    
    print(f"Analysis complete for {symbol}.")
    print(f"Method 1 (Fractal Pattern): {len(levels_1)} levels detected, Breakout: {has_breakout_1}")
    print(f"Method 2 (Window Shifting): {len(levels_2)} levels detected, Breakout: {has_breakout_2}")
    
    return results

def create_summary_plot(screened_list_1, screened_list_2, output_dir='plots', timestamp=None):
    """Create a summary plot comparing the two screening methods"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create sets for comparison
    set1 = set(screened_list_1)
    set2 = set(screened_list_2)
    
    # Find common stocks and unique stocks
    common = set1.intersection(set2)
    only_method1 = set1 - common
    only_method2 = set2 - common
    
    # Create data for the Venn diagram
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # Create a simple bar chart showing the counts
    labels = ['Method 1\n(Fractal Pattern)', 'Method 2\n(Window Shifting)', 'Both Methods']
    counts = [len(only_method1), len(only_method2), len(common)]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    bars = ax.bar(labels, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height}', ha='center', va='bottom', fontweight='bold')
    
    # Add title and labels
    ax.set_title('Breakout Pattern Detection Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Number of Stocks', fontsize=12, labelpad=10)
    
    # Add a grid
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Add stock symbols as text
    textstr = f"Common Stocks: {', '.join(sorted(common))}\n\n"
    textstr += f"Only Method 1: {', '.join(sorted(only_method1))}\n\n"
    textstr += f"Only Method 2: {', '.join(sorted(only_method2))}"
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
    plt.figtext(0.5, 0.01, textstr, ha='center', fontsize=9, bbox=props)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save figure
    filename = f'breakout_screening_summary_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved breakout screening summary to {filepath}")
    
    return filepath

def run_breakout_analysis(analysis_symbols=['AAPL'], 
                          scan_symbols=None,
                          scan_limit=10,
                          output_dir='plots',
                          start_date='2021-02-01',
                          timestamp=None):
    """
    Run a comprehensive breakout pattern analysis workflow.
    
    Parameters:
    -----------
    analysis_symbols : list
        List of stock symbols to analyze individually
    scan_symbols : list or None
        List of stock symbols to scan for breakout patterns. If None, uses S&P 500
    scan_limit : int
        Number of stocks to scan from the scan_symbols list
    output_dir : str
        Directory to save plot outputs
    start_date : str
        Start date for historical data in 'YYYY-MM-DD' format
    timestamp : str or None
        Timestamp to use in output filenames. If None, current timestamp is used
        
    Returns:
    --------
    dict
        Results of the analysis including individual analyses and screened stocks
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'individual_analyses': {},
        'screened_stocks': {
            'method1': [],
            'method2': []
        }
    }
    
    print("Starting Breakout Pattern Analysis")
    print("=================================")
    
    # 1. Analyze individual stocks
    for i, symbol in enumerate(analysis_symbols):
        print(f"\n{i+1}. Analyzing {symbol}...")
        print("-" * (len(f"{i+1}. Analyzing {symbol}...") + 1))
        
        analysis_result = analyze_single_stock(symbol, start_date, output_dir, timestamp)
        results['individual_analyses'][symbol] = analysis_result
    
    # 2. Screen for breakout patterns
    if scan_symbols is not None:
        stock_list = scan_symbols
    else:
        print("\nFetching S&P 500 stocks...")
        stock_list = get_sp500_stocks()
    
    print(f"\nScreening for breakout patterns...")
    print("-" * 35)
    
    screened_list_1, screened_list_2 = screen_for_breakouts(stock_list, start_date, scan_limit)
    
    results['screened_stocks']['method1'] = screened_list_1
    results['screened_stocks']['method2'] = screened_list_2
    
    print("\nStocks with breakout patterns (Method 1 - Fractal Pattern):")
    if screened_list_1:
        print(', '.join(screened_list_1))
    else:
        print("None found.")
    
    print("\nStocks with breakout patterns (Method 2 - Window Shifting):")
    if screened_list_2:
        print(', '.join(screened_list_2))
    else:
        print("None found.")
    
    # 3. Create summary plot
    if screened_list_1 or screened_list_2:
        print("\nCreating summary plot...")
        summary_plot = create_summary_plot(screened_list_1, screened_list_2, output_dir, timestamp)
        results['summary_plot'] = summary_plot
    
    print("\nBreakout Pattern Analysis Complete")
    print("=================================")
    
    return results

if __name__ == "__main__":
    # Example usage when script is run directly
    run_breakout_analysis(
        analysis_symbols=['AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOG', 'AMZN', 'META', 'TSM', 'NFLX', 'WMT', 'ORCL', 'IBM', 'CSCO', 'CMCSA', 'QCOM'],
        scan_limit=20,
        output_dir='plots/breakouts',
        start_date='2020-01-01'
    )
   