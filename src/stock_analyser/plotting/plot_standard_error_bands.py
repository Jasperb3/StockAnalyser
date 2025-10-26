import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from matplotlib.ticker import FuncFormatter
from scipy.stats import linregress
from stock_analyser.utils.constants import FONT_FAMILY

# Set the style for all plots
plt.style.use('seaborn-v0_8-darkgrid')

def compute_regression_line(prices, window):
    """
    Compute the rolling regression end-points over the given window.
    """
    prices_array = prices.to_numpy()
    reg_line = np.full(len(prices_array), np.nan)
    for i in range(window - 1, len(prices_array)):
        y = prices_array[i - window + 1:i + 1]
        x = np.arange(window)
        if np.any(np.isnan(y)):
            continue
        slope, intercept, _, _, _ = linregress(x, y)
        reg_line[i] = intercept + slope * (window - 1)
    return pd.Series(reg_line, index=prices.index)


def compute_standard_error(prices, reg_line, window):
    """
    Compute the standard error of prices from the regression line.
    """
    prices_array = prices.to_numpy()
    se = np.full(len(prices_array), np.nan)
    for i in range(window - 1, len(prices_array)):
        actual = prices_array[i - window + 1:i + 1]
        x = np.arange(window)
        if np.any(np.isnan(actual)):
            continue
        slope, intercept, _, _, _ = linregress(x, actual)
        predicted = intercept + slope * x
        residuals = actual - predicted
        se[i] = np.sqrt(np.sum(residuals**2) / (window - 2))
    return pd.Series(se, index=prices.index)

def standard_error_bands(df, price_col='Close', window=21, smooth=3):
    """
    Calculate Standard Error Bands for the given dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLCV data
    price_col : str
        Column name for price data (default: 'Close')
    window : int
        Window size for regression calculation (default: 21)
    smooth : int
        Smoothing window for regression line (default: 3)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional SEB columns
    """
    prices = df[price_col]
    reg = compute_regression_line(prices, window)
    reg_smooth = reg.rolling(smooth).mean()
    se = compute_standard_error(prices, reg, window)

    upper_band = reg_smooth + 2 * se
    lower_band = reg_smooth - 2 * se

    result = df.copy()
    result['SEB_Mid'] = reg_smooth
    result['SEB_Upper'] = upper_band
    result['SEB_Lower'] = lower_band
    return result

def money_formatter(x, pos):
    """Format y-axis labels as currency"""
    return f'${x:,.2f}'

def plot_standard_error_bands(ticker, period='6mo', output_dir='plots', timestamp=None):
    """
    Plot Standard Error Bands for a given ticker.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    period : str
        Period for historical data (default: '6mo')
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
    
    # Download data
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            print(f"❌ No data available for {ticker}")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading data for {ticker}: {str(e)}")
        return None
    
    # Calculate Standard Error Bands
    try:
        df = standard_error_bands(df)
    except Exception as e:
        print(f"❌ Error calculating Standard Error Bands for {ticker}: {str(e)}")
        return None
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 9), dpi=360)
    
    # Format the date axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    if len(df) > 120:  # More than 4 months of data
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    else:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    
    # Format the price axis
    ax.yaxis.set_major_formatter(FuncFormatter(money_formatter))
    
    # Plot the data
    ax.plot(df.index, df['Close'], label='Close Price', color='#1f77b4', linewidth=2)
    ax.plot(df.index, df['SEB_Mid'], label='Smoothed Regression (SMA)', 
            color='#ff7f0e', linewidth=1.5)
    ax.plot(df.index, df['SEB_Upper'], label='Upper Band (+2 SE)', 
            linestyle='--', color='#2ca02c', linewidth=1.5)
    ax.plot(df.index, df['SEB_Lower'], label='Lower Band (-2 SE)', 
            linestyle='--', color='#d62728', linewidth=1.5)
    
    # Fill between bands
    ax.fill_between(df.index, df['SEB_Lower'], df['SEB_Upper'], 
                    color='lightblue', alpha=0.2, label='Standard Error Channel')
    
    # Add title and labels
    ax.set_title(f'Standard Error Bands Analysis: {ticker}', 
                fontsize=16, fontweight='bold', pad=20, fontfamily=FONT_FAMILY)
    ax.set_xlabel('Date', fontsize=12, labelpad=10, fontfamily=FONT_FAMILY)
    ax.set_ylabel('Price', fontsize=12, labelpad=10, fontfamily=FONT_FAMILY)
    
    # Improve the legend
    ax.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=10)
    
    # Rotate date labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add a grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add parameter info text box
    textstr = f'Window: 21 periods\nSmoothing: 3 periods\nBands: ±2 Standard Errors'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, fontfamily=FONT_FAMILY)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    filename = f'{ticker}_{period}_standard_error_bands_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    
    # Save the plot
    plt.savefig(filepath, bbox_inches='tight', dpi=360, transparent=False)
    plt.close(fig)
    
    print(f"📊 Standard Error Bands chart saved to {filepath}")
    return filepath

def analyze_seb_signals(df):
    """
    Analyze Standard Error Bands for trading signals.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with SEB data
        
    Returns:
    --------
    dict
        Dictionary containing signal analysis
    """
    if 'SEB_Upper' not in df.columns or 'SEB_Lower' not in df.columns:
        return {"error": "Standard Error Bands not calculated"}
    
    latest_close = df['Close'].iloc[-1]
    latest_upper = df['SEB_Upper'].iloc[-1]
    latest_lower = df['SEB_Lower'].iloc[-1]
    latest_mid = df['SEB_Mid'].iloc[-1]
    
    # Determine position relative to bands
    if pd.isna(latest_upper) or pd.isna(latest_lower):
        position = "Insufficient data"
        signal = "No signal"
    elif latest_close > latest_upper:
        position = "Above upper band (Potential overbought)"
        signal = "Bearish divergence watch"
    elif latest_close < latest_lower:
        position = "Below lower band (Potential oversold)"
        signal = "Bullish reversal watch"
    elif latest_close > latest_mid:
        position = "Above regression line (Bullish bias)"
        signal = "Uptrend continuation"
    else:
        position = "Below regression line (Bearish bias)"
        signal = "Downtrend continuation"
    
    # Calculate band width (volatility indicator)
    if not pd.isna(latest_upper) and not pd.isna(latest_lower):
        band_width = ((latest_upper - latest_lower) / latest_mid) * 100
        volatility = "High" if band_width > 10 else "Medium" if band_width > 5 else "Low"
    else:
        band_width = None
        volatility = "Unknown"
    
    return {
        "position": position,
        "signal": signal,
        "volatility": volatility,
        "band_width_pct": round(band_width, 2) if band_width else None,
        "current_price": round(latest_close, 2),
        "upper_band": round(latest_upper, 2) if not pd.isna(latest_upper) else None,
        "lower_band": round(latest_lower, 2) if not pd.isna(latest_lower) else None,
        "regression_line": round(latest_mid, 2) if not pd.isna(latest_mid) else None
    }

if __name__ == "__main__":
    # Example usage
    plot_path = plot_standard_error_bands('AAPL', '6mo', 'test_plots')
    if plot_path:
        print(f"Plot saved successfully to: {plot_path}")