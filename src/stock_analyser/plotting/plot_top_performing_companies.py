import os
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt


def get_industry_from_ticker(ticker: str):
    company_ticker = yf.Ticker(ticker)
    industry = company_ticker.info['industry']
    print(f"Industry: {industry}")
    return industry


def get_top_performing_companies(ticker: str):
    company_ticker = yf.Ticker(ticker)
    industry_key = company_ticker.info['industryKey']
    print(f"Industry key: {industry_key}")

    top_performing_companies = yf.Industry(industry_key).top_performing_companies
    print(f"Top performing companies in industry dataframe:\n{top_performing_companies.to_markdown()}")

    return top_performing_companies


def top_performing_bar_chart(ticker: str, output_dir: str, timestamp: str):
    print("Plotting top performing companies in industry")

    top_performing_companies = get_top_performing_companies(ticker)

    top_performing_companies_tickers = top_performing_companies.index.tolist()
    print(f"Top performing companies in industry: {top_performing_companies_tickers}")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    bar_width = 0.4
    indices = np.arange(len(top_performing_companies))

    # Plot YTD Return on ax1 (left side) *before* the bars
    ax1.plot(indices, top_performing_companies['ytd return'], color='red', marker='o', linestyle='-', linewidth=2, label='YTD Return', zorder=2)
    ax1.set_ylabel('YTD Return (%)')  # Set y-label for ax1
    ax1.set_xlabel('Stock Symbol') # x label on the return axis

    # Plot last prices on ax2 (right side)
    ax2 = ax1.twinx()  # Create ax2 *before* plotting
    bars1 = ax2.bar(indices, top_performing_companies[' last price'], bar_width, label='Last Price', color='skyblue', zorder=1)

    # Plot target prices on top of last prices, also on ax2
    bars2 = ax2.bar(indices, top_performing_companies['target price'] - top_performing_companies[' last price'], bar_width, bottom=top_performing_companies[' last price'], label='Target Price', color='lightgreen', zorder=1)

    ax2.set_ylabel('Price (USD)')  # Set y-label for ax2
    industry = get_industry_from_ticker(ticker)
    ax2.set_title(f"Top performing companies in the {industry} industry")
    ax2.set_xticks(indices)
    ax2.set_xticklabels(top_performing_companies_tickers)

    # Combine legends
    bars = [bars1, bars2]
    labels = [bar.get_label() for bar in bars]
    # Add YTD return line to the legend
    lines, line_labels = ax1.get_legend_handles_labels()
    ax2.legend(bars + lines, labels + line_labels, loc='upper right')

    file_path = os.path.join(output_dir, f"{ticker}_{timestamp}_top_performing_companies_bar_chart.png")
    plt.tight_layout()
    plt.savefig(file_path)

    print(f"Bar chart saved to {file_path}")

    return file_path


def top_performing_dumbbell_chart(ticker: str, output_dir: str, timestamp: str):
    df = get_top_performing_companies(ticker)

    print(f"Columns in dataframe:\n{df.columns.tolist()}")

    # Replace NaN target prices with the last price to avoid plotting issues
    df['target price'] = df['target price'].fillna(df[' last price'])

    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define y positions for each stock
    y_positions = np.arange(len(df))

    # Plot the horizontal lines (dumbbells)
    for i in range(len(df)):
        row = df.iloc[i]
        ax.plot([row[' last price'], row['target price']], [y_positions[i]]*2, color='gray', linewidth=2, zorder=1)

    # Plot markers for last price and target price
    ax.scatter(df[' last price'], y_positions, color='skyblue', s=100, label='Last Price ($)', zorder=2)
    ax.scatter(df['target price'], y_positions, color='lightgreen', s=100, label='Target Price ($)', zorder=2)

    # Annotate with YTD return
    n = len(df)
    y_offset = 0.1
    for i in range(n):
        row = df.iloc[i]
        if i == n - 1:
            va = 'top'
            y_pos = y_positions[i] - y_offset
        else:
            va = 'bottom'
            y_pos = y_positions[i] + y_offset
        ax.text((row[' last price'] + row['target price']) / 2,
                y_pos,
                f"YTD:\n{row['ytd return']:.2%}",
                va=va, ha='center', fontsize=9, color='red')

    # Formatting the plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df.index)

    # Annotate last and target prices
    for i in range(len(df)):
      row = df.iloc[i]
      y_offset = 0.075  # Adjust this value as needed
      # Calculate a dynamic x_offset based on the difference between last and target price
      price_diff = abs(row['target price'] - row[' last price'])
      min_offset = 0.1 # minimum offset
      x_offset = max(min_offset, price_diff * 0.1)   # Adjust the 0.1 multiplier as needed

      # Annotate last price, shifting left if too close to target price
      if row[' last price'] < row['target price']:
          ha_last = 'right'
          x_last_pos = row[' last price'] - x_offset
      else:
          ha_last = 'left'
          x_last_pos = row[' last price'] + x_offset

      ax.text(x_last_pos, y_positions[i] - y_offset, f"{row[' last price']:.2f}", va='top', ha=ha_last, fontsize=8, color='skyblue')

      # Annotate target price, shifting right if too close to last price

      if row['target price'] > row[' last price']:
          ha_target = 'left'
          x_target_pos = row['target price'] + x_offset
      else:
          ha_target = 'right'
          x_target_pos = row['target price'] - x_offset
      ax.text(x_target_pos, y_positions[i] - y_offset, f"{row['target price']:.2f}", va='top', ha=ha_target, fontsize=8, color='lightgreen')

    ax.set_xlabel("Price (USD)")
    industry = get_industry_from_ticker(ticker)
    ax.set_title(f"Top performing companies in the {industry} industry")
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    file_path = os.path.join(output_dir, f"{ticker}_{timestamp}_top_performing_companies_dumbbell_chart.png")
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    print(f"Dumbbell chart saved to {file_path}")

    return file_path





if __name__ == "__main__":
    # TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    TIMESTAMP = "20250303_120000"
    PLOTS_DIR = "sandboxing"
    os.makedirs(PLOTS_DIR, exist_ok=True)
    # top_performing_bar_chart("NVDA", PLOTS_DIR, TIMESTAMP)
    top_performing_dumbbell_chart("GOOG", PLOTS_DIR, TIMESTAMP)
