import os
import numpy as np
import yfinance as yf
import mplfinance as mpf
from datetime import datetime
import matplotlib.pyplot as plt
from stock_analyser.utils.constants import FONT_FAMILY

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

    company_ticker = yf.Ticker(symbol)
    company_name = company_ticker.info.get('displayName') if company_ticker.info.get('displayName') else company_ticker.info.get('shortName')
    
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
    filepath = f'{output_dir}/{symbol}_squeeze_momentum_{timestamp}.png'
    
    # Get signal information
    signals = check_squeeze_signals(df)
    
    # Create title with signal information
    title = f'{company_name} Squeeze Momentum Indicator'
    subtitle = f"Status: {signals['status']} | Momentum: {signals['momentum']}"
    
    # Plot and save
    fig, axes = mpf.plot(
        ohlc,
        type='candle',
        style=s,
        addplot=apds,
        volume=True,
        figsize=(16, 12),
        # title=title,
        panel_ratios=(6, 3, 2),
        main_panel=0,      # Main panel is panel 0
        volume_panel=2,    # Volume is panel 2
        returnfig=True
    )

    # Add title
    axes[0].set_title(title, fontsize=24, fontfamily=FONT_FAMILY)
    
    # Add subtitle
    axes[0].text(0.5, 0.96, subtitle, transform=fig.transFigure, ha='center', fontsize=12, fontfamily=FONT_FAMILY)
    
    # Add legend for the momentum bars
    axes[0].text(0.05, 0.05, 'Momentum Bars:', transform=axes[1].transAxes, fontsize=9, fontfamily=FONT_FAMILY)
    axes[0].text(0.05, 0.02, 'Green/Lime: Bullish | Red/Maroon: Bearish', transform=axes[1].transAxes, fontsize=9, fontfamily=FONT_FAMILY)
    
    # Add legend for the squeeze markers
    axes[0].text(0.65, 0.05, 'Squeeze Status:', transform=axes[1].transAxes, fontsize=9, fontfamily=FONT_FAMILY)
    axes[0].text(0.65, 0.02, 'Black X: Squeeze ON | Gray X: Squeeze OFF', transform=axes[1].transAxes, fontsize=9, fontfamily=FONT_FAMILY)
    
    # Add signal information
    signal_text = ""
    if signals['long']:
        signal_text = "LONG SIGNAL DETECTED!"
    elif signals['short']:
        signal_text = "SHORT SIGNAL DETECTED!"
    
    if signal_text:
        axes[0].text(0.5, 0.93, signal_text, transform=fig.transFigure, ha='center', 
                    fontsize=14, fontweight='bold', color='red' if signals['short'] else 'green', fontfamily=FONT_FAMILY)
    
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


if __name__ == "__main__":
    # Example usage when script is run directly
    from stock_analyser.utils.constants import OUTPUT_DIR, TIMESTAMP
    results = analyze_single_stock(
        symbol='PLTR',
        output_dir=OUTPUT_DIR,
        timestamp=TIMESTAMP,
        # start_date='2020-01-01'
    )

    file_path = results.get('plot_path')
    # print(file_path)
    print(f"Results: {results}")