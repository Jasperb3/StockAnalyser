import os
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from stock_analyser.utils.constants import FONT_FAMILY


def plot_competitor_prices(ticker_list: list[str], output_dir: str, timestamp: str):
    # print("Plotting competitor prices")

    competitor_data = pd.DataFrame()  # Initialize an empty DataFrame

    for competitor in ticker_list:
        company_stock = yf.Ticker(competitor)
        if competitor.lower() in ['n/a', 'na', 'none', 'null', 'unknown', 'unavailable', 'missing'] or not company_stock:
            continue
        history = company_stock.history(period="1y")
        if history.empty:
            continue
        # Reset index to make 'Date' a column, and ensure consistent columns
        history = history.reset_index()
        if 'Date' in history.columns and 'Close' in history.columns:
            competitor_data[competitor] = history.set_index('Date')['Close']

    company_name = yf.Ticker(ticker_list[0]).info['shortName']

    # Create a new figure explicitly
    plt.figure(figsize=(10, 5))
    if not competitor_data.empty:
        plt.plot(competitor_data) # Plot all columns against the index
        plt.title(f"Stock price of {company_name}'s competitors over the last year", fontfamily=FONT_FAMILY)
        plt.xlabel('Date', fontfamily=FONT_FAMILY)
        plt.ylabel('Adjusted Close Stock Price ($)', fontfamily=FONT_FAMILY)
        plt.legend(competitor_data.columns) # Use column names for the legend

        file_path = os.path.join(output_dir, f"{ticker_list[0]}_{timestamp}_competitor_prices.png")
        plt.savefig(file_path, dpi=300)

        print(f"Plot saved to {file_path}")

        return file_path
    

if __name__ == "__main__":
    TIMESTAMP = "20250305_120000"

    PLOTS_DIR = "/home/j/ai/crewAI/finance/stock_analyser/sandboxing"

    plot_competitor_prices(["NVDA", "GOOG", "TSLA", "MSFT", "AMZN", "META", "NFLX", "ORCL", "IBM"], PLOTS_DIR, TIMESTAMP)