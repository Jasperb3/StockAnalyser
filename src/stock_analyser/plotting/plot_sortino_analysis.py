import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from typing import Union, List, Optional
from matplotlib.axes import Axes


today = datetime.today().strftime('%Y-%m-%d')
one_year_ago = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
start_of_decade = datetime(2020, 1, 1).strftime('%Y-%m-%d')


output_dir = "/home/j/ai/crewAI/finance/stock_analyser/plots/sortino"
time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")



def sortino_ratio(returns: Union[List[float], np.ndarray, pd.Series], 
                  target: float = 0.0, 
                  annualise: bool = False, 
                  periods_per_year: int = 252) -> float:
    """
    Calculate the Sortino Ratio of a return series.

    Parameters:
    - returns: array-like (list, np.array, or pd.Series) of periodic returns
    - target: Minimum Acceptable Return (e.g., 0.0 for 0%)
    - annualise: Whether to annualise the result
    - periods_per_year: 252 for daily, 12 for monthly, etc.

    Returns:
    - float: Sortino Ratio
    """
    returns = np.asarray(returns)
    downside_returns = returns[returns < target]
    downside_deviation = np.sqrt(np.mean((downside_returns - target) ** 2))
    excess_return = np.mean(returns) - target
    
    if downside_deviation == 0:
        return np.inf if excess_return > 0 else -np.inf

    sortino = excess_return / downside_deviation
    if annualise:
        sortino *= np.sqrt(periods_per_year)
    
    return sortino


def sharpe_ratio(returns: Union[List[float], np.ndarray, pd.Series], 
                risk_free_rate: float = 0.0, 
                annualise: bool = False, 
                periods_per_year: int = 252) -> float:
    """
    Compute the Sharpe Ratio from a return series.
    
    Parameters:
    - returns: array-like (list, np.array, or pd.Series) of periodic returns
    - risk_free_rate: The risk-free rate of return
    - annualise: Whether to annualise the result
    - periods_per_year: 252 for daily, 12 for monthly, etc.
    
    Returns:
    - float: Sharpe Ratio
    """
    returns = np.asarray(returns)  # Ensure it's a NumPy array
    excess_return = np.mean(returns) - risk_free_rate
    std_dev = np.std(returns)

    if np.isclose(std_dev, 0):
        return np.inf if excess_return > 0 else -np.inf

    sharpe = excess_return / std_dev
    if annualise:
        sharpe *= np.sqrt(periods_per_year)

    return sharpe


def rolling_sortino(returns: pd.Series, 
                   window: int = 60, 
                   target: float = 0.0, 
                   periods_per_year: int = 252) -> pd.Series:
    """
    Calculate rolling Sortino ratio for a series of returns.
    
    Parameters:
    - returns: pandas Series of periodic returns
    - window: Rolling window size
    - target: Minimum Acceptable Return (e.g., 0.0 for 0%)
    - periods_per_year: 252 for daily, 12 for monthly, etc.
    
    Returns:
    - pd.Series: Rolling Sortino ratios
    """
    return returns.rolling(window).apply(
        lambda x: sortino_ratio(x, target=target, annualise=True, periods_per_year=periods_per_year),
        raw=False
    )

def download_price_data(ticker: str) -> pd.Series:
    """
    Download historical price data for a given ticker.
    
    Parameters:
    - ticker: Stock symbol
    
    Returns:
    - pd.Series: Adjusted close prices
    """
    data = yf.download(ticker, start=start_of_decade, end=today, auto_adjust=False)
    prices = data["Adj Close"]
    return prices


def get_company_name(ticker: str) -> str:
    """
    Get the company name for a given ticker symbol.
    
    Parameters:
    - ticker: Stock symbol
    
    Returns:
    - str: Company name or the ticker if name not found
    """
    stock = yf.Ticker(ticker)
    return stock.info.get("displayName", stock.info.get("longName", stock.info.get("shortName", ticker)))


def calculate_drawdown(prices: pd.Series) -> pd.Series:
    """
    Calculate the drawdown series from a price series.
    
    Parameters:
    - prices: Series of prices
    
    Returns:
    - pd.Series: Drawdown series as percentage of peak
    """
    running_max = prices.cummax()
    drawdown = (prices - running_max) / running_max
    return drawdown


def rolling_volatility(returns: pd.Series, window: int = 60) -> pd.Series:
    """
    Calculate rolling annualized volatility.
    
    Parameters:
    - returns: Series of returns
    - window: Rolling window size
    
    Returns:
    - pd.Series: Rolling annualized volatility
    """
    return returns.rolling(window).std() * np.sqrt(252)  # Annualised


def rolling_return(returns: pd.Series, window: int = 60) -> pd.Series:
    """
    Calculate rolling annualized return.
    
    Parameters:
    - returns: Series of returns
    - window: Rolling window size
    
    Returns:
    - pd.Series: Rolling annualized return
    """
    return returns.rolling(window).mean() * 252  # Annualised


def add_event_annotations(ax: Axes, y_position: Optional[float] = None, fontsize: int = 11) -> None:
    """
    Add vertical lines and annotations for significant events using text symbols
    that are compatible with DejaVu Sans font.
    
    Parameters:
    - ax: Matplotlib axis to add annotations to
    - y_position: Vertical position for annotations (if None, calculated automatically)
    - fontsize: Font size for annotations
    """
    events = {
        "CV": "2020-03-15",  # COVID Crash
        "UA": "2022-02-24",  # Russia Invades Ukraine
        "TR": "2025-04-03"   # Trump Tariffs Announced
    }
    
    # Full labels for each event
    event_labels = {
        "CV": "COVID",
        "UA": "Ukraine War",
        "TR": "Tariffs"
    }
    
    # Calculate default y-position if not specified (90% of the way up the y-axis)
    if y_position is None:
        y_min, y_max = ax.get_ylim()
        y_position = y_min + 0.9 * (y_max - y_min)
    
    for label, date_str in events.items():
        date = pd.to_datetime(date_str)
        ax.axvline(x=date, linestyle='dotted', color='black', alpha=0.5)
        
        # # Add marker symbol at the top of the line
        # ax.annotate(label, 
        #            xy=(date, y_position),
        #            xytext=(0, 15),
        #            textcoords='offset points',
        #            ha='center',
        #            va='bottom',
        #            fontsize=fontsize,
        #            fontweight='bold',
        #            color='black',
        #            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
        #            zorder=5)
        
        # Add a small text label below
        ax.annotate(event_labels[label], 
                   xy=(date, y_position),
                   xytext=(0, -155),
                   textcoords='offset points',
                   ha='center',
                   va='top',
                   fontsize=fontsize-2,
                   color='black',
                   alpha=0.8)


def create_sortino_price_plot(ax1: Axes, rolling_sr: pd.Series, prices: pd.Series) -> None:
    """
    Create the Sortino Ratio vs Price plot.
    
    Parameters:
    - ax1: Primary axis for Sortino ratio
    - rolling_sr: Rolling Sortino ratio series
    - prices: Price series to display
    """
    ax2 = ax1.twinx()

    ax1.plot(rolling_sr, color='blue', label='60-Day Rolling Sortino', alpha=0.9)
    ax2.plot(prices, color='orange', alpha=0.6, label='Adj Close Price')

    ax1.axhline(y=1.0, linestyle='--', color='grey', label='Sortino = 1.0')
    add_event_annotations(ax1)

    ax1.set_ylabel("Sortino", color='blue')
    ax2.set_ylabel("Price ($)", color='orange')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax1.set_title("Rolling Sortino Ratio vs Adjusted Close Price")

    # Merge legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')


def create_volatility_return_plot(ax_vol: Axes, rolling_vol: pd.Series, rolling_ret: pd.Series) -> None:
    """
    Create the Volatility vs Return plot.
    
    Parameters:
    - ax_vol: Primary axis for volatility
    - rolling_vol: Rolling volatility series
    - rolling_ret: Rolling return series
    """
    ax_ret = ax_vol.twinx()

    ax_vol.plot(rolling_vol, color='red', label='Volatility (60d)')
    ax_ret.plot(rolling_ret, color='green', label='Return (60d)')
    add_event_annotations(ax_vol)

    ax_vol.set_ylabel("Volatility (%)", color='red')
    ax_ret.set_ylabel("Return (%)", color='green')
    ax_vol.tick_params(axis='y', labelcolor='red')
    ax_ret.tick_params(axis='y', labelcolor='green')
    ax_vol.set_title("Rolling Volatility vs Rolling Return")

    # Merge legends
    vol_line, vol_label = ax_vol.get_legend_handles_labels()
    ret_line, ret_label = ax_ret.get_legend_handles_labels()
    ax_vol.legend(vol_line + ret_line, vol_label + ret_label, loc='upper right')
    ax_vol.grid(True)


def create_drawdown_plot(ax: Axes, drawdown: pd.Series) -> None:
    """
    Create the Drawdown plot.
    
    Parameters:
    - ax: Matplotlib axis
    - drawdown: Drawdown series as percentage
    """
    ax.plot(drawdown, color='purple', label='Drawdown')
    ax.axhline(y=-20, linestyle='--', color='black', alpha=0.5, label='Pain point (-20%)')
    add_event_annotations(ax)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown from Peak")
    ax.legend()
    ax.grid(True)
    ax.set_xlabel("Date")


def plot_sortino_analysis(ticker: str) -> None:
    """
    Generate a comprehensive risk analysis dashboard for a stock.
    
    Creates a three-panel figure showing:
    1. Rolling Sortino ratio vs Price
    2. Rolling Volatility vs Rolling Return
    3. Drawdown from peak
    
    Parameters:
    - ticker: Stock symbol to analyze
    """
    prices = download_price_data(ticker)
    returns_series = prices.pct_change().dropna()
    company_name = get_company_name(ticker)
    file_name = f"{output_dir}/{ticker}_risk_dashboard_{time_stamp}.png"

    # Calculate metrics
    rolling_sr = rolling_sortino(returns_series, window=60)
    drawdown = calculate_drawdown(prices) * 100
    rolling_vol = rolling_volatility(returns_series) * 100
    rolling_ret = rolling_return(returns_series) * 100

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True, gridspec_kw={'hspace': 0.3})

    # Create each panel using the dedicated functions
    create_sortino_price_plot(axs[0], rolling_sr, prices)
    create_volatility_return_plot(axs[1], rolling_vol, rolling_ret)
    create_drawdown_plot(axs[2], drawdown)

    # Final title and layout
    fig.suptitle(f"{company_name} Risk-Adjusted Performance Dashboard (2020–)", fontsize=14, y=0.945)
    plt.tight_layout()

    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    print(f"Dashboard saved as {file_name}")


if __name__ == "__main__":
    ticker = "BRK-B"
    plot_sortino_analysis(ticker)





    