import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from stock_analyser.utils.constants import FONT_FAMILY

# Set the style for all plots
plt.style.use('seaborn-v0_8-darkgrid')

def Supertrend(df, atr_period, multiplier):
    
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # calculate ATR
    price_diffs = [high - low, 
                   high - close.shift(), 
                   close.shift() - low]
    true_range = pd.concat(price_diffs, axis=1)
    true_range = true_range.abs().max(axis=1)
    # default ATR calculation in supertrend indicator
    atr = true_range.ewm(alpha=1/atr_period,min_periods=atr_period).mean() 
    
    # HL2 is simply the average of high and low prices
    hl2 = (high + low) / 2
    # upperband and lowerband calculation
    # notice that final bands are set to be equal to the respective bands
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    
    # Create final bands series with same index as df
    final_upperband = pd.Series(index=df.index, dtype=float)
    final_lowerband = pd.Series(index=df.index, dtype=float)
    
    # Initialize first values
    final_upperband.iloc[0] = upperband.iloc[0]
    final_lowerband.iloc[0] = lowerband.iloc[0]
    
    # initialize Supertrend column to True
    supertrend = pd.Series([True] * len(df), index=df.index)
    
    for i in range(1, len(df)):
        # Set initial values for current position
        final_upperband.iloc[i] = upperband.iloc[i]
        final_lowerband.iloc[i] = lowerband.iloc[i]
        
        # if current close price crosses above upperband
        if close.iloc[i] > final_upperband.iloc[i-1]:
            supertrend.iloc[i] = True
        # if current close price crosses below lowerband
        elif close.iloc[i] < final_lowerband.iloc[i-1]:
            supertrend.iloc[i] = False
        # else, the trend continues
        else:
            supertrend.iloc[i] = supertrend.iloc[i-1]
            
            # adjustment to the final bands
            if supertrend.iloc[i] == True and final_lowerband.iloc[i] < final_lowerband.iloc[i-1]:
                final_lowerband.iloc[i] = final_lowerband.iloc[i-1]
            if supertrend.iloc[i] == False and final_upperband.iloc[i] > final_upperband.iloc[i-1]:
                final_upperband.iloc[i] = final_upperband.iloc[i-1]

        # to remove bands according to the trend direction
        if supertrend.iloc[i] == True:
            final_upperband.iloc[i] = np.nan
        else:
            final_lowerband.iloc[i] = np.nan
    
    return pd.DataFrame({
        'Supertrend': supertrend,
        'Final Lowerband': final_lowerband,
        'Final Upperband': final_upperband
    }, index=df.index)
    
def money_formatter(x, pos):
    """Format y-axis labels as currency"""
    return f'${x:,.2f}'

def plot_supertrend(df, filename='supertrend_plot.png', output_dir='plots'):
    # Create a figure with a specific size and DPI
    fig, ax = plt.subplots(figsize=(16, 9), dpi=360)
    
    # Format the date axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    # Format the price axis
    ax.yaxis.set_major_formatter(FuncFormatter(money_formatter))
    
    # Plot the data with enhanced styling
    ax.plot(df.index, df['Close'], linewidth=2, color='#1f77b4', label='Close Price')
    ax.plot(df.index, df['Final Lowerband'], color='#2ca02c', linewidth=1.5, 
            label='Support (Buy)', alpha=0.8)
    ax.plot(df.index, df['Final Upperband'], color='#d62728', linewidth=1.5, 
            label='Resistance (Sell)', alpha=0.8)
    
    # Fill the area between price and bands
    ax.fill_between(df.index, df['Close'], df['Final Lowerband'], 
                    where=(df['Final Lowerband'] > 0), color='#2ca02c', alpha=0.1)
    ax.fill_between(df.index, df['Close'], df['Final Upperband'], 
                    where=(df['Final Upperband'] > 0), color='#d62728', alpha=0.1)
    
    # Add title and labels with better styling
    symbol = filename.split('_')[0]
    ax.set_title(f'Supertrend Analysis: {symbol}', fontsize=16, fontweight='bold', pad=20, fontfamily=FONT_FAMILY)
    ax.set_xlabel('Date', fontsize=12, labelpad=10, fontfamily=FONT_FAMILY)
    ax.set_ylabel('Price', fontsize=12, labelpad=10, fontfamily=FONT_FAMILY)
    
    # Improve the legend
    ax.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=10)
    
    # Rotate date labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout to make room for rotated x-labels
    plt.tight_layout()
    
    # Add a grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a text box with period and multiplier info if available in the filename
    if '_period' in filename and '_mult' in filename:
        try:
            period = filename.split('_period')[1].split('_')[0]
            mult = filename.split('_mult')[1].split('_')[0]
            textstr = f'ATR Period: {period}\nMultiplier: {mult}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.80, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props, fontfamily=FONT_FAMILY)
        except:
            pass
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    # Save with tight layout and transparent background
    plt.savefig(filepath, bbox_inches='tight', dpi=360, transparent=False)
    plt.close(fig)
    print(f"Saved plot to {filepath}")

def backtest_supertrend(df, investment):
    is_uptrend = df['Supertrend']
    close = df['Close']
    
    # initial condition
    in_position = False
    equity = investment
    commission = 5
    share = 0
    entry = []
    exit = []
    
    for i in range(2, len(df)):
        # if not in position & price is on uptrend -> buy
        if not in_position and is_uptrend.iloc[i]:
            share = math.floor(equity / close.iloc[i] / 100) * 100
            equity -= share * close.iloc[i]
            entry.append((i, close.iloc[i]))
            in_position = True
        # if in position & price is not on uptrend -> sell
        elif in_position and not is_uptrend.iloc[i]:
            equity += share * close.iloc[i] - commission
            exit.append((i, close.iloc[i]))
            in_position = False
    
    # if still in position -> sell all share at the last price
    if in_position and len(df) > 0:
        last_idx = len(df) - 1
        equity += share * close.iloc[last_idx] - commission
    
    earning = equity - investment
    roi = round(earning/investment*100, 2)
    print(f'Earning from investing $100k is ${round(earning,2)} (ROI = {roi}%)')
    return entry, exit, roi

def plot_backtest(df, entry, exit, filename='backtest_plot.png', output_dir='plots'):
    # Create a figure with a specific size and DPI
    fig, ax = plt.subplots(figsize=(16, 9), dpi=360)
    
    # Format the date axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    # Format the price axis
    ax.yaxis.set_major_formatter(FuncFormatter(money_formatter))
    
    # Plot the price data
    ax.plot(df.index, df['Close'], linewidth=2, color='#1f77b4', label='Close Price')
    ax.plot(df.index, df['Final Lowerband'], color='#2ca02c', linewidth=1.5, 
            label='Support (Buy)', alpha=0.8)
    ax.plot(df.index, df['Final Upperband'], color='#d62728', linewidth=1.5, 
            label='Resistance (Sell)', alpha=0.8)
    
    # Fill the area between price and bands
    ax.fill_between(df.index, df['Close'], df['Final Lowerband'], 
                    where=(df['Final Lowerband'] > 0), color='#2ca02c', alpha=0.1)
    ax.fill_between(df.index, df['Close'], df['Final Upperband'], 
                    where=(df['Final Upperband'] > 0), color='#d62728', alpha=0.1)
    
    # Plot entry and exit points with enhanced styling
    for e in entry:
        ax.scatter(df.index[e[0]], e[1], marker='^', color='green', s=120, 
                  edgecolors='darkgreen', linewidth=1, zorder=5, label='_nolegend_')
    
    for e in exit:
        ax.scatter(df.index[e[0]], e[1], marker='v', color='red', s=120, 
                  edgecolors='darkred', linewidth=1, zorder=5, label='_nolegend_')
    
    # Add custom legend entries for entry and exit points
    if entry:
        ax.scatter([], [], marker='^', color='green', s=120, edgecolors='darkgreen', 
                  linewidth=1, label='Buy Signal')
    if exit:
        ax.scatter([], [], marker='v', color='red', s=120, edgecolors='darkred', 
                  linewidth=1, label='Sell Signal')
    
    # Add title and labels with better styling
    symbol = filename.split('_')[0]
    ax.set_title(f'Supertrend Backtest: {symbol}', fontsize=16, fontweight='bold', pad=20, fontfamily=FONT_FAMILY)
    ax.set_xlabel('Date', fontsize=12, labelpad=10, fontfamily=FONT_FAMILY)
    ax.set_ylabel('Price', fontsize=12, labelpad=10, fontfamily=FONT_FAMILY)
    
    # Improve the legend
    ax.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=10)
    
    # Rotate date labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout to make room for rotated x-labels
    plt.tight_layout()
    
    # Add a grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Calculate ROI for annotation
    if len(df) > 0:
        is_uptrend = df['Supertrend']
        close = df['Close']
        
        # Calculate ROI (simplified from backtest_supertrend)
        in_position = False
        equity = 100000  # Initial investment
        commission = 5
        share = 0
        
        for i in range(2, len(df)):
            if not in_position and is_uptrend.iloc[i]:
                share = math.floor(equity / close.iloc[i] / 100) * 100
                equity -= share * close.iloc[i]
                in_position = True
            elif in_position and not is_uptrend.iloc[i]:
                equity += share * close.iloc[i] - commission
                in_position = False
        
        if in_position and len(df) > 0:
            last_idx = len(df) - 1
            equity += share * close.iloc[last_idx] - commission
        
        earning = equity - 100000
        roi = round(earning/100000*100, 2)
        
        # Add ROI annotation
        roi_text = f'ROI: {roi}%\nTrades: {len(entry)}'
        props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7)
        ax.text(0.05, 0.80, roi_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props, fontfamily=FONT_FAMILY)
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    # Save with tight layout and transparent background
    plt.savefig(filepath, bbox_inches='tight', dpi=360, transparent=False)
    plt.close(fig)
    print(f"Saved plot to {filepath}")

def get_sp500_stocks():
    try:
        payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        return payload[0]['Symbol'].values.tolist()
    except Exception as e:
        print(f"Error fetching S&P 500 stocks: {str(e)}")
        return []

def find_supertrend_stocks(stock_list, atr_period, atr_multiplier, start_date='2010-01-01'):
    supertrend_stocks = []
    total = len(stock_list)
    
    print(f"Processing {total} stocks...")
    
    # loop through each symbol
    for i, symbol in enumerate(stock_list):
        try:
            print(f"Processing {symbol} ({i+1}/{total})...", end='\r')
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date)
            if len(df) < 2:  # Need at least 2 data points
                continue
            supertrend = Supertrend(df, atr_period, atr_multiplier)
            if len(supertrend) >= 2 and not supertrend['Supertrend'].iloc[-2] and supertrend['Supertrend'].iloc[-1]:
                supertrend_stocks.append(symbol)
        except Exception as e:
            print(f"\nError processing {symbol}: {str(e)}")
    
    print("\nProcessing complete.")
    return supertrend_stocks

def find_optimal_parameter(df):
    # predefine several parameter sets
    atr_period = [7, 8, 9, 10]
    atr_multiplier = [1.0, 1.5, 2.0, 2.5, 3.0]

    roi_list = []
    total_combinations = len(atr_period) * len(atr_multiplier)
    
    print(f"Testing {total_combinations} parameter combinations...")
    
    # for each period and multiplier, perform backtest
    count = 0
    for period, multiplier in [(x,y) for x in atr_period for y in atr_multiplier]:
        count += 1
        try:
            print(f"Testing period={period}, multiplier={multiplier} ({count}/{total_combinations})...", end='\r')
            new_df = df.copy()
            supertrend = Supertrend(new_df, period, multiplier)
            new_df = new_df.join(supertrend)
            new_df = new_df.iloc[period:]  # Skip the first 'period' rows
            if len(new_df) > 0:
                entry, exit, roi = backtest_supertrend(new_df, 100000)
                roi_list.append((period, multiplier, roi))
        except Exception as e:
            print(f"\nError with period={period}, multiplier={multiplier}: {str(e)}")
    
    print("\nParameter testing complete.")
    
    if not roi_list:
        print("No valid parameter combinations found.")
        return (10, 3.0, 0.0)  # Default values if no valid combinations
        
    results_df = pd.DataFrame(roi_list, columns=['ATR_period','Multiplier','ROI'])
    print("\nResults:")
    print(results_df)
    
    # return the best parameter set
    best_params = max(roi_list, key=lambda x:x[2])
    print(f"\nBest parameters: ATR Period={best_params[0]}, Multiplier={best_params[1]}, ROI={best_params[2]}%")
    return best_params

def analyze_single_stock(symbol, atr_period, atr_multiplier, start_date='2020-01-01', 
                         output_dir='plots', timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    print(f"Downloading data for {symbol} from {start_date}...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date)
    
    if len(df) < 2:
        print(f"Not enough data for {symbol}")
        return None, [], [], 0.0
    
    print(f"Calculating Supertrend for {symbol}...")
    supertrend = Supertrend(df, atr_period, atr_multiplier)
    df = df.join(supertrend)
    
    # Plot supertrend with parameter info in filename
    print(f"Plotting Supertrend for {symbol}...")
    supertrend_filename = f'{symbol}_supertrend_period{atr_period}_mult{atr_multiplier}_{timestamp}.png'
    plot_supertrend(df, supertrend_filename, output_dir)
    
    # Backtest
    print(f"Running backtest for {symbol}...")
    entry, exit, roi = backtest_supertrend(df, 100000)
    
    # Plot backtest
    print(f"Plotting backtest results for {symbol}...")
    backtest_filename = f'{symbol}_backtest_period{atr_period}_mult{atr_multiplier}_{timestamp}.png'
    plot_backtest(df, entry, exit, backtest_filename, output_dir)
    
    return df, entry, exit, roi

def run_supertrend_analysis(analysis_symbols=['AAPL'], 
                           scan_symbols=None, 
                           scan_limit=10,
                           optimization_symbol='TSLA',
                           output_dir='plots',
                           start_date='2020-01-01',
                           timestamp=None):
    """
    Run a comprehensive supertrend analysis workflow.
    
    Parameters:
    -----------
    analysis_symbols : list
        List of stock symbols to analyze with default parameters
    scan_symbols : list or None
        List of stock symbols to scan for supertrend signals. If None, uses S&P 500
    scan_limit : int
        Number of stocks to scan from the scan_symbols list
    optimization_symbol : str
        Symbol to use for parameter optimization
    output_dir : str
        Directory to save plot outputs
    start_date : str
        Start date for historical data in 'YYYY-MM-DD' format
    timestamp : str or None
        Timestamp to use in output filenames. If None, current timestamp is used
        
    Returns:
    --------
    dict
        Results of the analysis including ROIs, optimal parameters, and signal stocks
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Default parameters
    atr_period = 10
    atr_multiplier = 3.0
    
    results = {
        'analysis_results': {},
        'supertrend_signals': [],
        'optimal_parameters': None,
        'optimization_roi': None
    }
    
    print("Starting Supertrend Analysis")
    print("===========================")
    
    # 1. Analyze specified stocks with default parameters
    for i, symbol in enumerate(analysis_symbols):
        print(f"\n{i+1}. Analyzing {symbol}...")
        print("-" * (len(f"{i+1}. Analyzing {symbol}...") + 1))
        
        df, entry, exit, roi = analyze_single_stock(
            symbol, atr_period, atr_multiplier, start_date, output_dir, timestamp
        )
        
        if df is not None:
            print(f"{symbol} analysis complete. ROI: {roi}%")
            results['analysis_results'][symbol] = {
                'roi': roi,
                'entry_points': len(entry),
                'exit_points': len(exit)
            }
    
    # 2. Find stocks with supertrend signal
    if scan_symbols is not None:
        stock_list = scan_symbols
    else:
        print("\nFetching S&P 500 stocks...")
        stock_list = get_sp500_stocks()
    
    # Use a limited list for scanning
    if scan_limit and scan_limit < len(stock_list):
        demo_list = stock_list[:scan_limit]
    else:
        demo_list = stock_list
        
    print(f"\nScanning for supertrend signals...")
    print("-" * 35)
    print(f"Using a sample of {len(demo_list)} stocks")
    
    supertrend_stocks = find_supertrend_stocks(demo_list, atr_period, atr_multiplier, start_date)
    
    print("\nStocks with supertrend buy signal:")
    if supertrend_stocks:
        for s in supertrend_stocks:
            print(s, end=', ')
        results['supertrend_signals'] = supertrend_stocks
    else:
        print("None found in the sample.")
    print("\n")
    
    # 3. Parameter optimization
    if optimization_symbol:
        print(f"\nOptimizing parameters for {optimization_symbol}...")
        print("-" * (len(f"Optimizing parameters for {optimization_symbol}...") + 1))
        
        ticker = yf.Ticker(optimization_symbol)
        opt_df = ticker.history(start=start_date)
        
        if len(opt_df) >= 2:
            optimal_param = find_optimal_parameter(opt_df)
            results['optimal_parameters'] = {
                'atr_period': optimal_param[0],
                'multiplier': optimal_param[1],
                'roi': optimal_param[2]
            }
            
            # 4. Analyze with optimal parameters
            print(f"\nAnalyzing {optimization_symbol} with optimal parameters...")
            print("-" * (len(f"Analyzing {optimization_symbol} with optimal parameters...") + 1))
            
            _, _, _, opt_roi = analyze_single_stock(
                optimization_symbol, optimal_param[0], optimal_param[1], 
                start_date, output_dir, f"optimal_{timestamp}"
            )
            
            results['optimization_roi'] = opt_roi
            print(f"{optimization_symbol} analysis with optimal parameters complete.")
        else:
            print(f"Not enough data for {optimization_symbol}")
    
    # Create a summary plot of all analyzed stocks
    if results['analysis_results']:
        create_summary_plot(results, output_dir, timestamp)
    
    print("\nSupertrend Analysis Complete")
    print("===========================")
    
    return results

def create_summary_plot(results, output_dir, timestamp):
    """Create a summary bar chart of ROIs for all analyzed stocks"""
    analysis_results = results['analysis_results']
    
    if not analysis_results:
        return
    
    # Extract ROIs and symbols
    symbols = list(analysis_results.keys())
    rois = [analysis_results[symbol]['roi'] for symbol in symbols]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    
    # Create bars with colors based on ROI (green for positive, red for negative)
    colors = ['#2ca02c' if roi >= 0 else '#d62728' for roi in rois]
    bars = ax.bar(symbols, rois, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add value labels on top of bars
    for bar, roi in zip(bars, rois):
        height = bar.get_height()
        label_pos = height + 0.5 if height >= 0 else height - 5
        ax.text(bar.get_x() + bar.get_width()/2., label_pos,
                f'{roi:.2f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                fontweight='bold')
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add title and labels
    ax.set_title('Supertrend Strategy ROI Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Stock Symbol', fontsize=12, labelpad=10)
    ax.set_ylabel('Return on Investment (%)', fontsize=12, labelpad=10)
    
    # Add a grid for better readability
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Adjust layout
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'roi_summary_{timestamp}.png')
    
    # Save the plot
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved ROI summary plot to {filepath}")

if __name__ == "__main__":
    # Example usage when script is run directly
    run_supertrend_analysis(
        analysis_symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
        scan_limit=5,
        optimization_symbol='TSLA',
        output_dir='plots/supertrend',
        start_date='2020-01-01'
    )



