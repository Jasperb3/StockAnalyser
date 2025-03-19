import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

# --- Constants for Consistent Styling ---
FONT_FAMILY = 'DejaVu Sans'
TITLE_FONT_SIZE = 14
AXIS_LABEL_FONT_SIZE = 12
TICK_LABEL_FONT_SIZE = 10
LEGEND_FONT_SIZE = 10
LINE_WIDTH = 2
MARKER_SIZE = 6
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Blue, Orange, Green, Red, Purple
GRID_LINE_STYLE = '--'
GRID_ALPHA = 0.5

# --- Helper Functions ---

def format_currency(value, pos):
    """Formats currency values for y-axis labels."""
    if value >= 1e9:
        return f'${value/1e9:.1f}B'  # Format as billions
    elif value >= 1e6:
        return f'${value/1e6:.1f}M'  # Format as millions
    elif value >= 1e3:
        return f'${value/1e3:.0f}K'  # Format as thousands
    else:
        return f'${value:.2f}' # Format as is

def format_percentage(value, pos):
    """Formats percentage values for y-axis labels."""
    return f'{value:.1%}'

def apply_common_styling(ax, title):
    """Applies common styling to all charts."""
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight='bold', fontfamily=FONT_FAMILY)
    ax.set_xlabel('Year', fontsize=AXIS_LABEL_FONT_SIZE, fontfamily=FONT_FAMILY)
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(linestyle=GRID_LINE_STYLE, alpha=GRID_ALPHA)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(FONT_FAMILY)

def plot_revenue_trends(ticker: str, output_dir: Path, timestamp: str) -> Path:
    """
    Plot the revenue trends for a given stock ticker.
    """
    stock = yf.Ticker(ticker)
    financial_data = stock.financials
    revenue_data = financial_data.loc['Total Revenue']

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(revenue_data.index, revenue_data.values, marker='o', linestyle='-', color=COLORS[0],
            linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

    apply_common_styling(ax, f'{ticker} Total Revenue (USD)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_currency))

    # Add data label for the most recent year
    last_year = revenue_data.index[-1].year
    last_revenue = revenue_data.values[-1]
    ax.annotate(format_currency(last_revenue, None),  # Use the same formatting as the y-axis
                xy=(revenue_data.index[-1], last_revenue),
                xytext=(5, 5),  # Offset the text slightly
                textcoords='offset points',
                fontsize=TICK_LABEL_FONT_SIZE,
                fontfamily=FONT_FAMILY)

    plt.tight_layout()
    file_path = Path(f"{output_dir}/{ticker}_revenue_trends_{timestamp}.png")
    plt.savefig(file_path)
    return file_path


def plot_net_income_trends(ticker: str, output_dir: Path, timestamp: str) -> Path:
    """
    Plot the Net Income for a given stock ticker.
    """
    stock = yf.Ticker(ticker)
    financial_data = stock.financials
    net_income_data = financial_data.loc['Net Income']

    fig, ax = plt.subplots(figsize=(10, 5))

    # Separate positive and negative values
    positive_income = net_income_data.copy()
    negative_income = net_income_data.copy()
    positive_income[positive_income <= 0] = np.nan
    negative_income[negative_income > 0] = np.nan

    ax.plot(positive_income.index, positive_income.values, marker='o', linestyle='-', color='green',
            linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='Positive')
    ax.plot(negative_income.index, negative_income.values, marker='o', linestyle='-', color='red',
            linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='Negative')

    apply_common_styling(ax, f'{ticker} Net Income (USD)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_currency))
    ax.legend(fontsize=LEGEND_FONT_SIZE, loc='upper left')

    plt.tight_layout()
    file_path = Path(f"{output_dir}/{ticker}_net_income_trends_{timestamp}.png")
    plt.savefig(file_path)
    return file_path


def plot_gross_margin_trends(ticker: str, output_dir: Path, timestamp: str) -> Path:
    """
    Plot the Gross Margin for a given stock ticker.
    """
    stock = yf.Ticker(ticker)
    financial_data = stock.financials
    gross_profit_data = financial_data.loc['Gross Profit']
    revenue_data = financial_data.loc['Total Revenue']
    gross_margin_data = gross_profit_data / revenue_data

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gross_margin_data.index, gross_margin_data.values, marker='o', linestyle='-', color=COLORS[2],
            linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

    apply_common_styling(ax, f'{ticker} Gross Margin (%)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_percentage))

    plt.tight_layout()
    file_path = Path(f"{output_dir}/{ticker}_gross_margin_trends_{timestamp}.png")
    plt.savefig(file_path)
    return file_path


def plot_earnings_per_share_trends(ticker: str, output_dir: Path, timestamp: str) -> Path:
    """
    Plot the EPS trends for a given stock ticker.
    """
    stock = yf.Ticker(ticker)
    financial_data = stock.financials
    basic_eps_data = financial_data.loc['Basic EPS']
    diluted_eps_data = financial_data.loc['Diluted EPS']

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(basic_eps_data.index, basic_eps_data.values, marker='o', linestyle='-', color=COLORS[3],
            linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='Basic EPS')
    ax.plot(diluted_eps_data.index, diluted_eps_data.values, marker='o', linestyle='-', color=COLORS[4],
            linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='Diluted EPS')

    apply_common_styling(ax, f'{ticker} Earnings Per Share (USD)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_currency)) # EPS is currency
    ax.legend(fontsize=LEGEND_FONT_SIZE, loc='upper left')

    plt.tight_layout()
    file_path = Path(f"{output_dir}/{ticker}_earnings_per_share_trends_{timestamp}.png")
    plt.savefig(file_path)
    return file_path


def plot_free_cash_flow_trends(ticker: str, output_dir: Path, timestamp: str) -> Path:
    """
    Plot the Free Cash Flow for a given stock ticker.
    """
    stock = yf.Ticker(ticker)
    cash_flow_data = stock.cashflow
    free_cash_flow_data = cash_flow_data.loc['Free Cash Flow']

    fig, ax = plt.subplots(figsize=(10, 5))

    # Separate positive and negative values
    positive_fcf = free_cash_flow_data.copy()
    negative_fcf = free_cash_flow_data.copy()
    positive_fcf[positive_fcf <= 0] = np.nan
    negative_fcf[negative_fcf > 0] = np.nan

    ax.plot(positive_fcf.index, positive_fcf.values, marker='o', linestyle='-', color='green',
            linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='Positive')
    ax.plot(negative_fcf.index, negative_fcf.values, marker='o', linestyle='-', color='red',
            linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='Negative')


    apply_common_styling(ax, f'{ticker} Free Cash Flow (USD)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_currency))
    ax.legend(fontsize=LEGEND_FONT_SIZE, loc='upper left')

    plt.tight_layout()
    file_path = Path(f"{output_dir}/{ticker}_free_cash_flow_trends_{timestamp}.png")
    plt.savefig(file_path)
    return file_path

if __name__ == "__main__":
    ticker = 'SOUN'
    output_dir = Path('plots/fin_data')
    timestamp = '20250317_233000'

    revenue_plot = plot_revenue_trends(ticker, output_dir, timestamp)
    net_income_plot = plot_net_income_trends(ticker, output_dir, timestamp)
    gross_margin_plot = plot_gross_margin_trends(ticker, output_dir, timestamp)
    earnings_per_share_plot = plot_earnings_per_share_trends(ticker, output_dir, timestamp)
    free_cash_flow_plot = plot_free_cash_flow_trends(ticker, output_dir, timestamp)

    print(f"Revenue plot saved to {revenue_plot}")
    print(f"Net income plot saved to {net_income_plot}")
    print(f"Gross margin plot saved to {gross_margin_plot}")
    print(f"Earnings per share plot saved to {earnings_per_share_plot}")
    print(f"Free cash flow plot saved to {free_cash_flow_plot}")













