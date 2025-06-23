# Re-import necessary libraries after execution state reset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from matplotlib.gridspec import GridSpec
from stock_analyser.utils.constants import FONT_FAMILY

def get_prices(ticker, start_date, end_date):
    """
    Fetches adjusted closing prices for a ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        pandas.Series: Series of adjusted closing prices
    """
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date, auto_adjust=False)
    return df['Adj Close']

def get_name(ticker):
    """
    Fetches the display name for a ticker.
    
    Args:
        ticker (str): Stock ticker symbol
    
    Returns:
        str: Display name
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    return info.get('displayName', info.get('shortName', info.get('longName', ticker)))

def calculate_mswing(prices, length1, length2):
    """
    Calculate Mswing indicator values.
    
    Args:
        prices (pandas.Series): Series of prices
        length1 (int): First lookback period
        length2 (int): Second lookback period
        
    Returns:
        numpy.ndarray: Mswing indicator values
    """
    mswing = np.zeros(len(prices))
    for i in range(len(prices)):
        if i >= length2:
            mswing[i] = ((prices.iloc[i] - prices.iloc[i - length1]) / prices.iloc[i - length1] * 100) + \
                        ((prices.iloc[i] - prices.iloc[i - length2]) / prices.iloc[i - length2] * 100)
        else:
            mswing[i] = np.nan  # Not enough data
    return mswing

def ema(series, span):
    """
    Calculate Exponential Moving Average.
    
    Args:
        series (numpy.ndarray or pandas.Series): Data series
        span (int): EMA period
        
    Returns:
        numpy.ndarray: EMA values
    """
    return pd.Series(series).ewm(span=span, adjust=False).mean().to_numpy()

def fetch_data(ticker, index_ticker='^GSPC', period_days=365):
    """
    Fetch stock and index data.
    
    Args:
        ticker (str): Stock ticker symbol
        index_ticker (str): Index ticker symbol (default: S&P 500)
        period_days (int): Number of days of data to fetch
        
    Returns:
        tuple: (stock_prices, index_prices, company_name)
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=period_days)).strftime("%Y-%m-%d")
    
    stock_prices = get_prices(ticker, start_date, end_date)
    index_prices = get_prices(index_ticker, start_date, end_date)
    
    # Ensure both series have the same index
    common_dates = stock_prices.index.intersection(index_prices.index)
    stock_prices = stock_prices.loc[common_dates]
    index_prices = index_prices.loc[common_dates]
    
    return stock_prices, index_prices

def create_mswing_plot(ticker, index_ticker='^GSPC', length1=20, length2=50, ema_period=9, output_dir=None, timestamp=None):
    """
    Main function to create Mswing indicator plot with relative strength.
    
    Args:
        ticker (str): Stock ticker symbol
        index_ticker (str): Index ticker symbol (default: S&P 500)
        length1 (int): First lookback period for Mswing
        length2 (int): Second lookback period for Mswing
        ema_period (int): EMA period for smoothing
        output_dir (str): Directory to save the plot
        
    Returns:
        str: Path to the saved plot
    """
    # Fetch data
    stock_prices, index_prices = fetch_data(ticker, index_ticker)
    company_name = get_name(ticker)
    index_name = get_name(index_ticker)
    
    # Calculate Mswing for stock and index
    mswing_stock = calculate_mswing(stock_prices, length1, length2)
    mswing_index = calculate_mswing(index_prices, length1, length2)
    
    # Calculate EMAs
    ema_mswing_stock = ema(mswing_stock, ema_period)
    ema_mswing_index = ema(mswing_index, ema_period)
    
    # Calculate Relative Strength
    relative_strength = mswing_stock - mswing_index
    ema_relative_strength = ema(relative_strength, ema_period)
    
    # Create the plot
    fig = plt.figure(figsize=(14, 10), dpi=300)
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1])
    
    # Plot 1: Stock Mswing
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(stock_prices.index, mswing_stock, label=f"{ticker} Mswing", color="blue", linewidth=1.5, alpha=0.7)
    ax1.plot(stock_prices.index, ema_mswing_stock, label=f"EMA({ema_period}) of {ticker} Mswing", color="darkblue", linewidth=1.5)
    ax1.axhline(0, color="gray", linestyle="dashed", alpha=0.5)
    ax1.set_title(f"{company_name} Mswing Indicator (Periods: {length1}, {length2})", fontsize=14, fontweight='bold', fontfamily=FONT_FAMILY)
    ax1.set_ylabel("Mswing Value", fontsize=12, fontfamily=FONT_FAMILY)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(stock_prices.index, mswing_stock, 0, where=(mswing_stock > 0), color='green', alpha=0.2)
    ax1.fill_between(stock_prices.index, mswing_stock, 0, where=(mswing_stock < 0), color='red', alpha=0.2)
    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    
    # Plot 2: Index Mswing
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(stock_prices.index, mswing_index, label=f"{index_name} Mswing", color="orange", linewidth=1.5, alpha=0.7)
    ax2.plot(stock_prices.index, ema_mswing_index, label=f"EMA({ema_period}) of {index_name} Mswing", color="darkorange", linewidth=1.5)
    ax2.axhline(0, color="gray", linestyle="dashed", alpha=0.5)
    ax2.set_title(f"{index_name} Mswing Indicator", fontsize=14, fontweight='bold', fontfamily=FONT_FAMILY)
    ax2.set_ylabel("Mswing Value", fontsize=12, fontfamily=FONT_FAMILY)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.fill_between(stock_prices.index, mswing_index, 0, where=(mswing_index > 0), color='green', alpha=0.2)
    ax2.fill_between(stock_prices.index, mswing_index, 0, where=(mswing_index < 0), color='red', alpha=0.2)
    ax2.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    
    # Plot 3: Relative Strength
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(stock_prices.index, relative_strength, label="Relative Strength", color="purple", linewidth=1.5, alpha=0.8)
    ax3.plot(stock_prices.index, ema_relative_strength, label=f"EMA({ema_period}) of Relative Strength", color="darkviolet", linewidth=1.5)
    ax3.axhline(0, color="gray", linestyle="dashed", alpha=0.5)
    ax3.set_title(f"Mswing Relative Strength: {company_name} vs {index_name}", fontsize=14, fontweight='bold', fontfamily=FONT_FAMILY)
    ax3.set_xlabel("Date", fontsize=12, fontfamily=FONT_FAMILY)
    ax3.set_ylabel("Relative Strength Value", fontsize=12, fontfamily=FONT_FAMILY)
    ax3.legend(loc="upper left", fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.fill_between(stock_prices.index, relative_strength, 0, where=(relative_strength > 0), color='green', alpha=0.3, label="Outperforming")
    ax3.fill_between(stock_prices.index, relative_strength, 0, where=(relative_strength < 0), color='red', alpha=0.3, label="Underperforming")
    
    # Add annotations for the last value
    last_date = stock_prices.index[-1]
    last_rs = relative_strength[-1]
    rs_status = "Outperforming" if last_rs > 0 else "Underperforming"
    rs_color = "green" if last_rs > 0 else "red"
    ax3.annotate(f"{rs_status} ({last_rs:.2f})", 
                xy=(last_date, last_rs),
                xytext=(last_date - pd.Timedelta(days=10), last_rs + (0.5 if last_rs > 0 else -0.5)),
                arrowprops=dict(facecolor=rs_color, shrink=0.05, width=1.5, headwidth=8),
                fontsize=10,
                fontweight='bold',
                color=rs_color)
    
    # Add interpretation text at the bottom
    interpretation = (
        f"Mswing Indicator compares current price to {length1}-day and {length2}-day historical prices.\n"
        f"Relative Strength shows if {company_name} is outperforming (above 0) or underperforming (below 0) the {index_name} index."
    )
    fig.text(0.5, 0.01, interpretation, ha='center', fontsize=10, fontfamily=FONT_FAMILY, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
           
    filename = f"{output_dir}/{ticker}_mswing_relative_strength_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    
    return filename

# Script execution entry point
if __name__ == "__main__":
    
    ticker = "NVDA"  # Default ticker
    index_ticker = "^GSPC"  # Default index (S&P 500)
    output_dir = "/home/j/ai/crewAI/finance/stock_analyser/plots/mswing"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    create_mswing_plot(ticker, index_ticker, output_dir=output_dir, timestamp=timestamp)