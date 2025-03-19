import os
import numpy as np
import yfinance as yf
import mplfinance as mpf
from datetime import datetime
import matplotlib.pyplot as plt

# Set the style for all plots
plt.style.use('seaborn-v0_8-darkgrid')

def calculate_squeeze_momentum(df, length=20, mult=2, length_KC=20, mult_KC=1.5):
    """
    Calculate the Squeeze Momentum Indicator (LazyBear)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLC data
    length : int
        Length for Bollinger Bands calculation
    mult : float
        Multiplier for Bollinger Bands
    length_KC : int
        Length for Keltner Channel calculation
    mult_KC : float
        Multiplier for Keltner Channel
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added Squeeze Momentum indicators
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Calculate Bollinger Bands
    m_avg = df['Close'].rolling(window=length).mean()
    m_std = df['Close'].rolling(window=length).std(ddof=0)
    df['upper_BB'] = m_avg + mult * m_std
    df['lower_BB'] = m_avg - mult * m_std

    # Calculate true range
    df['tr0'] = abs(df["High"] - df["Low"])
    df['tr1'] = abs(df["High"] - df["Close"].shift())
    df['tr2'] = abs(df["Low"] - df["Close"].shift())
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)

    # Calculate Keltner Channel
    range_ma = df['tr'].rolling(window=length_KC).mean()
    df['upper_KC'] = m_avg + range_ma * mult_KC
    df['lower_KC'] = m_avg - range_ma * mult_KC

    # Calculate momentum bar value
    highest = df['High'].rolling(window=length_KC).max()
    lowest = df['Low'].rolling(window=length_KC).min()
    m1 = (highest + lowest)/2
    df['value'] = (df['Close'] - (m1 + m_avg)/2)
    fit_y = np.array(range(0, length_KC))
    df['value'] = df['value'].rolling(window=length_KC).apply(lambda x: 
                                np.polyfit(fit_y, x, 1)[0] * (length_KC-1) + 
                                np.polyfit(fit_y, x, 1)[1], raw=True)

    # Check for 'squeeze'
    df['squeeze_on'] = (df['lower_BB'] > df['lower_KC']) & (df['upper_BB'] < df['upper_KC'])
    df['squeeze_off'] = (df['lower_BB'] < df['lower_KC']) & (df['upper_BB'] > df['upper_KC'])

    # Clean up intermediate columns
    df = df.drop(['tr0', 'tr1', 'tr2', 'tr'], axis=1)
    
    return df

def check_squeeze_signals(df):
    """
    Check for squeeze momentum signals in the data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with squeeze momentum indicators
        
    Returns:
    --------
    dict
        Dictionary with signal information
    """
    if len(df) < 2:
        return {'long': False, 'short': False, 'status': 'Not enough data'}
    
    # Buying window for long position:
    # 1. Black cross becomes gray (the squeeze is released)
    long_cond1 = (df['squeeze_off'].iloc[-2] == False) & (df['squeeze_off'].iloc[-1] == True) 
    # 2. Bar value is positive => the bar is light green
    long_cond2 = df['value'].iloc[-1] > 0
    enter_long = long_cond1 and long_cond2

    # Buying window for short position:
    # 1. Black cross becomes gray (the squeeze is released)
    short_cond1 = (df['squeeze_off'].iloc[-2] == False) & (df['squeeze_off'].iloc[-1] == True) 
    # 2. Bar value is negative => the bar is light red 
    short_cond2 = df['value'].iloc[-1] < 0
    enter_short = short_cond1 and short_cond2

    # Current squeeze status
    if df['squeeze_on'].iloc[-1]:
        status = "Squeeze ON (Consolidation)"
    else:
        status = "Squeeze OFF (Trending)"
    
    # Momentum direction
    if df['value'].iloc[-1] > 0:
        if df['value'].iloc[-1] > df['value'].iloc[-2]:
            momentum = "Increasing Bullish Momentum"
        else:
            momentum = "Decreasing Bullish Momentum"
    else:
        if df['value'].iloc[-1] < df['value'].iloc[-2]:
            momentum = "Increasing Bearish Momentum"
        else:
            momentum = "Decreasing Bearish Momentum"
    
    return {
        'long': enter_long,
        'short': enter_short,
        'status': status,
        'momentum': momentum,
        'value': df['value'].iloc[-1]
    }

def plot_squeeze_momentum(df, symbol, output_dir='plots', timestamp=None, lookback=100):
    """
    Plot the Squeeze Momentum Indicator
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with squeeze momentum indicators
    symbol : str
        Stock symbol
    output_dir : str
        Directory to save the plot
    timestamp : str or None
        Timestamp for the filename
    lookback : int
        Number of bars to show in the plot
        
    Returns:
    --------
    str
        Path to the saved plot
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use only the last 'lookback' rows for better visualization
    plot_df = df[-lookback:].copy()
    
    # Extract OHLC data
    ohlc = plot_df[['Open', 'High', 'Close', 'Low', 'Volume']]
    
    # Add colors for the 'value bar'
    colors = []
    for ind, val in enumerate(plot_df['value']):
        if ind == 0:  # Handle first bar
            prev_val = 0
        else:
            prev_val = plot_df['value'].iloc[ind-1]
            
        if val >= 0:
            color = 'green'
            if val > prev_val:
                color = 'lime'
        else:
            color = 'maroon'
            if val < prev_val:
                color = 'red'
        colors.append(color)

    # Create marker colors for squeeze status
    marker_colors = ['gray' if s else 'black' for s in plot_df['squeeze_off']]
    
    # Add subplots: 1. momentum bars, 2. squeeze markers
    apds = [
        mpf.make_addplot(plot_df['value'], panel=1, type='bar', color=colors, alpha=0.8, secondary_y=False),
        mpf.make_addplot([0] * len(plot_df), panel=1, type='scatter', marker='x', markersize=50, color=marker_colors, secondary_y=False)
    ]
    
    # Add Bollinger Bands and Keltner Channels
    apds.append(mpf.make_addplot(plot_df['upper_BB'], color='blue', alpha=0.3, linestyle='--'))
    apds.append(mpf.make_addplot(plot_df['lower_BB'], color='blue', alpha=0.3, linestyle='--'))
    apds.append(mpf.make_addplot(plot_df['upper_KC'], color='red', alpha=0.3, linestyle='--'))
    apds.append(mpf.make_addplot(plot_df['lower_KC'], color='red', alpha=0.3, linestyle='--'))
    
    # Set style
    mc = mpf.make_marketcolors(
        up='#2ca02c',
        down='#d62728',
        edge='inherit',
        wick='inherit',
        volume='inherit'
    )
    
    s = mpf.make_mpf_style(
        base_mpf_style='charles',
        marketcolors=mc,
        gridstyle='--',
        gridcolor='gray',
        gridaxis='both',
        y_on_right=False
    )
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set filename
    filename = f'{symbol}_squeeze_momentum_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    
    # Get signal information
    signals = check_squeeze_signals(df)
    
    # Create title with signal information
    title = f'{symbol} - Squeeze Momentum Indicator (LazyBear)'
    subtitle = f"Status: {signals['status']} | Momentum: {signals['momentum']}"
    
    # Plot and save
    fig, axes = mpf.plot(
        ohlc,
        type='candle',
        style=s,
        addplot=apds,
        volume=True,
        figsize=(16, 12),
        title=title,
        panel_ratios=(6, 3, 2),
        main_panel=0,      # Main panel is panel 0
        volume_panel=2,    # Volume is panel 2
        returnfig=True
    )
    
    # Add subtitle
    axes[0].text(0.5, 0.96, subtitle, transform=fig.transFigure, ha='center', fontsize=12)
    
    # Add legend for the momentum bars
    axes[0].text(0.05, 0.05, 'Momentum Bars:', transform=axes[1].transAxes, fontsize=9)
    axes[0].text(0.05, 0.02, 'Green/Lime: Bullish | Red/Maroon: Bearish', transform=axes[1].transAxes, fontsize=9)
    
    # Add legend for the squeeze markers
    axes[0].text(0.65, 0.05, 'Squeeze Status:', transform=axes[1].transAxes, fontsize=9)
    axes[0].text(0.65, 0.02, 'Black X: Squeeze ON | Gray X: Squeeze OFF', transform=axes[1].transAxes, fontsize=9)
    
    # Add signal information
    signal_text = ""
    if signals['long']:
        signal_text = "LONG SIGNAL DETECTED!"
    elif signals['short']:
        signal_text = "SHORT SIGNAL DETECTED!"
    
    if signal_text:
        axes[0].text(0.5, 0.93, signal_text, transform=fig.transFigure, ha='center', 
                    fontsize=14, fontweight='bold', color='red' if signals['short'] else 'green')
    
    # Save the figure
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"Saved Squeeze Momentum plot to {filepath}")
    return filepath

def screen_for_squeeze_signals(stock_list, length=20, mult=2, length_KC=20, mult_KC=1.5, start_date='2020-01-01'):
    """
    Screen a list of stocks for squeeze momentum signals
    
    Parameters:
    -----------
    stock_list : list
        List of stock symbols to screen
    length : int
        Length for Bollinger Bands calculation
    mult : float
        Multiplier for Bollinger Bands
    length_KC : int
        Length for Keltner Channel calculation
    mult_KC : float
        Multiplier for Keltner Channel
    start_date : str
        Start date for historical data
        
    Returns:
    --------
    dict
        Dictionary with long and short signal stocks
    """
    long_signals = []
    short_signals = []
    
    total = len(stock_list)
    print(f"Screening {total} stocks for Squeeze Momentum signals...")
    
    for i, symbol in enumerate(stock_list):
        try:
            print(f"Processing {symbol} ({i+1}/{total})...", end='\r')
            
            # Get stock data
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date)
            
            if len(df) < length_KC + 10:  # Need enough data for calculations
                continue
                
            # Calculate squeeze momentum indicators
            df = calculate_squeeze_momentum(df, length, mult, length_KC, mult_KC)
            
            # Check for signals
            signals = check_squeeze_signals(df)
            
            if signals['long']:
                long_signals.append(symbol)
            elif signals['short']:
                short_signals.append(symbol)
                
        except Exception as e:
            print(f"\nError processing {symbol}: {str(e)}")
    
    print("\nScreening complete.")
    return {'long': long_signals, 'short': short_signals}

def analyze_single_stock(symbol, length=20, mult=2, length_KC=20, mult_KC=1.5, 
                         start_date='2020-01-01', output_dir='plots', timestamp=None):
    """
    Analyze a single stock with the Squeeze Momentum Indicator
    
    Parameters:
    -----------
    symbol : str
        Stock symbol to analyze
    length : int
        Length for Bollinger Bands calculation
    mult : float
        Multiplier for Bollinger Bands
    length_KC : int
        Length for Keltner Channel calculation
    mult_KC : float
        Multiplier for Keltner Channel
    start_date : str
        Start date for historical data
    output_dir : str
        Directory to save the plot
    timestamp : str or None
        Timestamp for the filename
        
    Returns:
    --------
    dict
        Dictionary with analysis results
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Analyzing {symbol} with Squeeze Momentum Indicator...")
    
    # Get stock data
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date)
    
    if len(df) < length_KC + 10:
        print(f"Not enough data for {symbol}")
        return None
    
    # Calculate squeeze momentum indicators
    df = calculate_squeeze_momentum(df, length, mult, length_KC, mult_KC)
    
    # Check for signals
    signals = check_squeeze_signals(df)
    
    # Plot the indicator
    plot_path = plot_squeeze_momentum(df, symbol, output_dir, timestamp)
    
    # Return analysis results
    results = {
        'symbol': symbol,
        'signals': signals,
        'plot_path': plot_path,
        'parameters': {
            'length': length,
            'mult': mult,
            'length_KC': length_KC,
            'mult_KC': mult_KC
        }
    }
    
    print(f"Analysis complete for {symbol}.")
    print(f"Status: {signals['status']}")
    print(f"Momentum: {signals['momentum']}")
    if signals['long']:
        print("LONG SIGNAL DETECTED!")
    elif signals['short']:
        print("SHORT SIGNAL DETECTED!")
    
    return results

def create_summary_plot(results, output_dir='plots', timestamp=None):
    """
    Create a summary plot of squeeze momentum screening results
    
    Parameters:
    -----------
    results : dict
        Dictionary with screening results
    output_dir : str
        Directory to save the plot
    timestamp : str or None
        Timestamp for the filename
        
    Returns:
    --------
    str
        Path to the saved plot
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    long_signals = results.get('long', [])
    short_signals = results.get('short', [])
    
    if not long_signals and not short_signals:
        print("No signals to plot in summary.")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    
    # Create data for the plot
    labels = ['Long Signals', 'Short Signals']
    counts = [len(long_signals), len(short_signals)]
    colors = ['#2ca02c', '#d62728']
    
    # Create bars
    bars = ax.bar(labels, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height}', ha='center', va='bottom', fontweight='bold')
    
    # Add title and labels
    ax.set_title('Squeeze Momentum Signal Summary', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Number of Stocks', fontsize=12, labelpad=10)
    
    # Add a grid
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Add stock symbols as text
    textstr = f"Long Signals: {', '.join(sorted(long_signals))}\n\n"
    textstr += f"Short Signals: {', '.join(sorted(short_signals))}"
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
    plt.figtext(0.5, 0.01, textstr, ha='center', fontsize=9, bbox=props)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save figure
    filename = f'squeeze_momentum_summary_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved Squeeze Momentum summary to {filepath}")
    return filepath

def run_squeeze_momentum_analysis(analysis_symbols=['AAPL'], 
                                 scan_symbols=None,
                                 scan_limit=10,
                                 length=20, 
                                 mult=2, 
                                 length_KC=20, 
                                 mult_KC=1.5,
                                 output_dir='plots/squeeze_momentum',
                                 start_date='2020-01-01',
                                 timestamp=None):
    """
    Run a comprehensive Squeeze Momentum analysis workflow.
    
    Parameters:
    -----------
    analysis_symbols : list
        List of stock symbols to analyze individually
    scan_symbols : list or None
        List of stock symbols to scan for signals. If None, uses a default list
    scan_limit : int
        Number of stocks to scan from the scan_symbols list
    length : int
        Length for Bollinger Bands calculation
    mult : float
        Multiplier for Bollinger Bands
    length_KC : int
        Length for Keltner Channel calculation
    mult_KC : float
        Multiplier for Keltner Channel
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
            'long': [],
            'short': []
        }
    }
    
    print("Starting Squeeze Momentum Analysis")
    print("=================================")
    
    # 1. Analyze individual stocks
    for i, symbol in enumerate(analysis_symbols):
        print(f"\n{i+1}. Analyzing {symbol}...")
        print("-" * (len(f"{i+1}. Analyzing {symbol}...") + 1))
        
        analysis_result = analyze_single_stock(
            symbol, length, mult, length_KC, mult_KC, 
            start_date, output_dir, timestamp
        )
        
        if analysis_result:
            results['individual_analyses'][symbol] = analysis_result
    
    # 2. Screen for signals
    if scan_symbols is not None:
        stock_list = scan_symbols
    else:
        # Default to S&P 500 or a smaller list for demonstration
        default_stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'DJT', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V', 
                         'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'BAC', 'ADBE', 'CMCSA', 'XOM']
        stock_list = default_stocks
    
    # Use a limited list for scanning
    if scan_limit and scan_limit < len(stock_list):
        scan_list = stock_list[:scan_limit]
    else:
        scan_list = stock_list
    
    print(f"\nScreening for Squeeze Momentum signals...")
    print("-" * 40)
    print(f"Using a sample of {len(scan_list)} stocks")
    
    screened_results = screen_for_squeeze_signals(
        scan_list, length, mult, length_KC, mult_KC, start_date
    )
    
    results['screened_stocks'] = screened_results
    
    print("\nStocks with Long signals:")
    if screened_results['long']:
        print(', '.join(screened_results['long']))
    else:
        print("None found.")
    
    print("\nStocks with Short signals:")
    if screened_results['short']:
        print(', '.join(screened_results['short']))
    else:
        print("None found.")
    
    # 3. Create summary plot
    if screened_results['long'] or screened_results['short']:
        print("\nCreating summary plot...")
        summary_plot = create_summary_plot(screened_results, output_dir, timestamp)
        results['summary_plot'] = summary_plot
    
    print("\nSqueeze Momentum Analysis Complete")
    print("=================================")
    
    return results

if __name__ == "__main__":
    # Example usage when script is run directly
    run_squeeze_momentum_analysis(
        analysis_symbols=['NVDA'],
        scan_limit=20,
        output_dir='plots/squeeze_momentum',
        start_date='2020-01-01'
    )