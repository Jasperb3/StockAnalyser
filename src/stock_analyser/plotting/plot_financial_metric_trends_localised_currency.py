import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path
from stock_analyser.utils.currency_symbols import currency_symbols
from stock_analyser.utils.constants import FONT_FAMILY

# --- Constants for Consistent Styling ---
TITLE_FONT_SIZE = 14
AXIS_LABEL_FONT_SIZE = 12
TICK_LABEL_FONT_SIZE = 10
LEGEND_FONT_SIZE = 10
LINE_WIDTH = 2
MARKER_SIZE = 6
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
GRID_LINE_STYLE = '--'
GRID_ALPHA = 0.5

# --- Helper Functions ---

def apply_common_styling(ax, title):
    """Applies common styling to all charts."""
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight='bold', fontfamily=FONT_FAMILY)
    ax.set_xlabel('Year', fontsize=AXIS_LABEL_FONT_SIZE, fontfamily=FONT_FAMILY)
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(linestyle=GRID_LINE_STYLE, alpha=GRID_ALPHA)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(FONT_FAMILY)

def plot_revenue_trends(ticker: str, output_dir: Path, timestamp: str) -> Path:
    stock = yf.Ticker(ticker)
    try:
        financial_data = stock.financials
        revenue_data = financial_data.loc['Total Revenue']
        currency = stock.info['financialCurrency']
        currency_symbol = currency_symbols.get(currency, currency)
    except Exception as e:
        print(f"Error plotting revenue trends for {ticker}: {e}")
        return None

    def currency_formatter(value, pos):
        if value >= 1e9:
            return f'{currency_symbol}{value/1e9:.1f}B'
        elif value >= 1e6:
            return f'{currency_symbol}{value/1e6:.1f}M'
        elif value >= 1e3:
            return f'{currency_symbol}{value/1e3:.0f}K'
        else:
            return f'{currency_symbol}{value:.2f}'

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(revenue_data.index, revenue_data.values, marker='o', linestyle='-', color=COLORS[0],
            linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

    apply_common_styling(ax, f'{ticker} Total Revenue ({currency})')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(currency_formatter))

    last_year = revenue_data.index[-1].year
    last_revenue = revenue_data.values[-1]
    ax.annotate(currency_formatter(last_revenue, None),
                xy=(revenue_data.index[-1], last_revenue),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=TICK_LABEL_FONT_SIZE,
                fontfamily=FONT_FAMILY)

    plt.tight_layout()
    file_path = Path(f"{output_dir}/{ticker}_revenue_trends_{timestamp}_localised.png")
    plt.savefig(file_path, dpi=300)
    return file_path

def plot_net_income_trends(ticker: str, output_dir: Path, timestamp: str) -> Path:
    stock = yf.Ticker(ticker)
    try:
        financial_data = stock.financials
        net_income_data = financial_data.loc['Net Income']
        currency = stock.info['financialCurrency']
        currency_symbol = currency_symbols.get(currency, currency)
    except Exception as e:
        print(f"Error plotting net income trends for {ticker}: {e}")
        return None

    def currency_formatter(value, pos):
        if value >= 1e9:
            return f'{currency_symbol}{value/1e9:.1f}B'
        elif value >= 1e6:
            return f'{currency_symbol}{value/1e6:.1f}M'
        elif value >= 1e3:
            return f'{currency_symbol}{value/1e3:.0f}K'
        else:
            return f'{currency_symbol}{value:.2f}'

    fig, ax = plt.subplots(figsize=(10, 5))

    positive_income = net_income_data.copy()
    negative_income = net_income_data.copy()
    positive_income[positive_income <= 0] = np.nan
    negative_income[negative_income > 0] = np.nan

    ax.plot(positive_income.index, positive_income.values, marker='o', linestyle='-', color='green',
            linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='Positive')
    ax.plot(negative_income.index, negative_income.values, marker='o', linestyle='-', color='red',
            linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='Negative')

    apply_common_styling(ax, f'{ticker} Net Income ({currency})')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(currency_formatter))
    ax.legend(fontsize=LEGEND_FONT_SIZE, loc='upper left')

    plt.tight_layout()
    file_path = Path(f"{output_dir}/{ticker}_net_income_trends_{timestamp}_localised.png")
    plt.savefig(file_path, dpi=300)
    return file_path

def plot_gross_margin_trends(ticker: str, output_dir: Path, timestamp: str) -> Path:
    stock = yf.Ticker(ticker)
    try:
        financial_data = stock.financials
        gross_profit_data = financial_data.loc['Gross Profit']
        revenue_data = financial_data.loc['Total Revenue']
        gross_margin_data = gross_profit_data / revenue_data
    except Exception as e:
        print(f"Error plotting gross margin trends for {ticker}: {e}")
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gross_margin_data.index, gross_margin_data.values, marker='o', linestyle='-', color=COLORS[2],
            linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

    apply_common_styling(ax, f'{ticker} Gross Margin (%)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f'{val:.1%}'))

    plt.tight_layout()
    file_path = Path(f"{output_dir}/{ticker}_gross_margin_trends_{timestamp}_localised.png")
    plt.savefig(file_path, dpi=300)
    return file_path

def plot_earnings_per_share_trends(ticker: str, output_dir: Path, timestamp: str) -> Path:
    stock = yf.Ticker(ticker)
    try:
        financial_data = stock.financials
        basic_eps_data = financial_data.loc['Basic EPS']
        diluted_eps_data = financial_data.loc['Diluted EPS']
        currency = stock.info['financialCurrency']
        currency_symbol = currency_symbols.get(currency, currency)
    except Exception as e:
        print(f"Error plotting earnings per share trends for {ticker}: {e}")
        return None

    def currency_formatter(value, pos):
        if value >= 1e9:
            return f'{currency_symbol}{value/1e9:.1f}B'
        elif value >= 1e6:
            return f'{currency_symbol}{value/1e6:.1f}M'
        elif value >= 1e3:
            return f'{currency_symbol}{value/1e3:.0f}K'
        else:
            return f'{currency_symbol}{value:.2f}'

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(basic_eps_data.index, basic_eps_data.values, marker='o', linestyle='-', color=COLORS[3],
            linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='Basic EPS')
    ax.plot(diluted_eps_data.index, diluted_eps_data.values, marker='o', linestyle='-', color=COLORS[4],
            linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='Diluted EPS')

    apply_common_styling(ax, f'{ticker} Earnings Per Share ({currency})')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(currency_formatter))
    ax.legend(fontsize=LEGEND_FONT_SIZE, loc='upper left')

    plt.tight_layout()
    file_path = Path(f"{output_dir}/{ticker}_earnings_per_share_trends_{timestamp}_localised.png")
    plt.savefig(file_path, dpi=300)
    return file_path

def plot_free_cash_flow_trends(ticker: str, output_dir: Path, timestamp: str) -> Path:
    stock = yf.Ticker(ticker)
    try:
        cash_flow_data = stock.cashflow
        free_cash_flow_data = cash_flow_data.loc['Free Cash Flow']
        currency = stock.info['financialCurrency']
        currency_symbol = currency_symbols.get(currency, currency)
    except Exception as e:
        print(f"Error plotting free cash flow trends for {ticker}: {e}")
        return None

    def currency_formatter(value, pos):
        if value >= 1e9:
            return f'{currency_symbol}{value/1e9:.1f}B'
        elif value >= 1e6:
            return f'{currency_symbol}{value/1e6:.1f}M'
        elif value >= 1e3:
            return f'{currency_symbol}{value/1e3:.0f}K'
        else:
            return f'{currency_symbol}{value:.2f}'

    fig, ax = plt.subplots(figsize=(10, 5))

    positive_fcf = free_cash_flow_data.copy()
    negative_fcf = free_cash_flow_data.copy()
    positive_fcf[positive_fcf <= 0] = np.nan
    negative_fcf[negative_fcf > 0] = np.nan

    ax.plot(positive_fcf.index, positive_fcf.values, marker='o', linestyle='-', color='green',
            linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='Positive')
    ax.plot(negative_fcf.index, negative_fcf.values, marker='o', linestyle='-', color='red',
            linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='Negative')

    apply_common_styling(ax, f'{ticker} Free Cash Flow ({currency})')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(currency_formatter))
    ax.legend(fontsize=LEGEND_FONT_SIZE, loc='upper left')

    plt.tight_layout()
    file_path = Path(f"{output_dir}/{ticker}_free_cash_flow_trends_{timestamp}_localised.png")
    plt.savefig(file_path, dpi=300)
    return file_path

if __name__ == "__main__":
    ticker = 'MANU'
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
