import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from stock_analyser.utils.constants import FONT_FAMILY


# Fetch stock and S&P 500 (^GSPC) data
def fetch_data(ticker):

    stock = yf.Ticker(ticker)

    sp500 = yf.Ticker('^GSPC')

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    # stock prices data
    stock_prices = stock.history(start=start_date, end=end_date)
    # S&P 500 index data
    sp500_index_data = sp500.history(start=start_date, end=end_date)

    return stock_prices, sp500_index_data

def calculate_rs_line(stock_prices, sp500):
    # Calculate the RS Line by dividing stock closing price by S&P 500 closing price
    rs_line = stock_prices['Close'] / sp500['Close']
    return rs_line

def calculate_rsi(data, rsi_window, ma_window):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=rsi_window).mean()
    avg_loss = loss.rolling(window=rsi_window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Calculate 10-day moving average of RSI
    rsi_ma_10 = rsi.rolling(window=ma_window).mean()

    data['RSI_14'] = rsi
    data['RSI_10_MA'] = rsi_ma_10

    return data

def plot_data(stock_prices, rs_line, rsi, ticker, output_dir, timestamp):
    ma_list = [10, 20, 50, 100]

    fig = plt.figure(figsize=(12, 10), dpi=300)
    gs = GridSpec(4, 1, height_ratios=[4, 1, 3,2])

    # Plot stock stock price with line chart and moving averages
    ax_stock = fig.add_subplot(gs[0])
    ax_stock.plot(stock_prices.index, stock_prices['Close'], label='Close Price', color='black')
    for ma in ma_list:
        stock_ma = stock_prices['Close'].rolling(window=ma).mean()
        ax_stock.plot(stock_prices.index, stock_ma, label=f'{ma}-Day MA', alpha=0.5, linestyle='-')
    ax_stock.set_title(f'{ticker} Stock Price with Moving Averages', fontfamily=FONT_FAMILY, fontsize=18)
    ax_stock.set_ylabel('Stock Price (USD)', fontfamily=FONT_FAMILY)
    ax_stock.legend(loc='upper left')
    ax_stock.grid()
    ax_stock.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # Plot volume in grayscale
    ax_volume = fig.add_subplot(gs[1], sharex=ax_stock)
    ax_volume.bar(stock_prices.index, stock_prices['Volume'], color='gray', alpha=0.5)
    ax_volume.set_ylabel('Volume', fontfamily=FONT_FAMILY)
    ax_volume.grid()
    ax_volume.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # Plot RS Line
    ax_rs = fig.add_subplot(gs[2], sharex=ax_stock)
    rs_line_ma10 = rs_line.rolling(window=10).mean()
    ax_rs.plot(stock_prices.index, rs_line, label=f'RS Line ({ticker} vs S&P 500)', color='orange')
    ax_rs.plot(stock_prices.index, rs_line_ma10, label='10-Day MA', alpha=0.8, color='gray', linestyle='-')
    ax_rs.fill_between(stock_prices.index, rs_line, rs_line_ma10, where=(rs_line < rs_line_ma10), color='red', alpha=0.5, label='Below MA 10')
    ax_rs.fill_between(stock_prices.index, rs_line, rs_line_ma10, where=(rs_line > rs_line_ma10), color='green', alpha=0.5, label='Above MA 10')
    ax_rs.set_title(f'Relative Strength Line ({ticker} vs S&P 500)', fontfamily=FONT_FAMILY, fontsize=18)
    ax_rs.set_xlabel('Date', fontfamily=FONT_FAMILY)
    ax_rs.set_ylabel('RS Value', fontfamily=FONT_FAMILY)
    ax_rs.legend(loc='upper left')
    ax_rs.grid()
    
    # Plot RSI
    ax_rsi = fig.add_subplot(gs[3], sharex=ax_stock)
    ax_rsi.plot(stock_prices.index, stock_prices['RSI_14'], label='RSI (14-day)', color='darkblue')
    ax_rsi.plot(stock_prices.index, stock_prices['RSI_10_MA'], label='10-Day MA of RSI', color='red', alpha=0.5, linestyle='-')
    ax_rsi.axhline(y=30, color='red', linestyle='-', linewidth=1, label='Oversold (30)')
    ax_rsi.axhline(y=70, color='green', linestyle='-', linewidth=1, label='Overbought (70)')
    ax_rsi.set_title('Relative Strength Index (RSI)', fontfamily=FONT_FAMILY, fontsize=18)
    ax_rsi.set_xlabel('Date', fontfamily=FONT_FAMILY)
    ax_rsi.set_ylabel('RSI Value', fontfamily=FONT_FAMILY)
    ax_rsi.legend(loc='upper left')
    ax_rsi.grid()


    plt.tight_layout()

    file_path = f'{output_dir}/{ticker}_relative_strength_{timestamp}.png'
    plt.savefig(file_path, dpi=300)

    return file_path


# Main 
def plot_relative_strength(ticker, output_dir, timestamp):
    print(f'Plotting Relative Strength for {ticker}...')
    stock_prices, sp500_index_data = fetch_data(ticker)
    rs_line = calculate_rs_line(stock_prices, sp500_index_data)
    rsi = calculate_rsi(stock_prices, 14,10)
    file_path = plot_data(stock_prices, rs_line, rsi, ticker, output_dir, timestamp)
    print(f'Plot saved to {file_path}')
    return file_path

if __name__ == "__main__":
    from stock_analyser.utils.constants import TIMESTAMP, PLOTS_DIR
    ticker = 'APP'
    output_dir = PLOTS_DIR
    timestamp = TIMESTAMP
    plot_relative_strength(ticker, output_dir, timestamp)