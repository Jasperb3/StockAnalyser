import os
import re
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import mplfinance as mpf
from palettable.cartocolors.sequential import BluGrn_5, BluGrn_4, BluGrn_3, BluGrn_6, BluGrn_7
from stock_analyser.utils.constants import FONT_FAMILY


def plot_historical_stock_data(ticker: str, output_dir: str, timestamp: str):
    print("Plotting historical stock data")

    company_stock = yf.Ticker(ticker)

    # Fetch historical stock data
    stock_data_5_years = company_stock.history(period="5y")

    # Reset the index to make 'Date' a regular column
    stock_data_5_years = stock_data_5_years.reset_index()

    # Create a figure and axis
    plt.figure()
    plt.plot(stock_data_5_years['Date'], stock_data_5_years['Close'], label=ticker)

    # Add labels, title, and legend
    plt.xlabel('Year', fontfamily=FONT_FAMILY)
    plt.ylabel('Stock Price ($)', fontfamily=FONT_FAMILY)
    plt.title(f'Historical Stock Prices for {ticker}', fontfamily=FONT_FAMILY)
    plt.legend()

    file_path = os.path.join(output_dir, f"{ticker}_{timestamp}_5-year_stock_price_plot.png")
    plt.savefig(file_path, dpi=300)

    print(f"Plot saved to {file_path}")

    return file_path


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
    

def plot_anaylsts_recommendations(ticker: str, output_dir: str, timestamp: str):
    print("Plotting pie chart of analysts' recommendations")
    title = "Analyst Recommendations"

    company_ticker = yf.Ticker(ticker)

    try:
        # Fetch recommendations and handle potential errors
        recommendations_summary = company_ticker.recommendations_summary
        if recommendations_summary is None or recommendations_summary.empty:
            raise ValueError("No recommendation summary data available.")

        current_recommendations = recommendations_summary.drop(columns=['period']).reset_index(drop=True)
        current_recommendations = current_recommendations.iloc[0]
        current_recommendations = current_recommendations.astype(int)

        # Extract labels (keys) and values
        
        pattern = re.compile(r'(?<!^)(?=[A-Z])')
        labels = [pattern.sub(' ', label).lower().capitalize() for label in current_recommendations.index]
        recommendation_counts = np.array(current_recommendations.values)

        # Remove negative or zero values with warning
        mask = recommendation_counts > 0
        if not all(mask):
            print(f"{sum(~mask)} negative or zero values removed from pie chart")
            recommendation_counts = recommendation_counts[mask]
            labels = [label for i, label in enumerate(labels) if mask[i]]

        n = len(mask)
        print(f"Number of valid recommendations: {n}")

        palettes = {
            3: BluGrn_3,
            4: BluGrn_4,
            5: BluGrn_5,
            6: BluGrn_6,
            7: BluGrn_7,
        }
        if n not in palettes:
            print(f"Warning: No palette available for n={n}, using n=7")
            n=7
        
        colors = palettes.get(n).mpl_colors
        # Handle the case where no valid values remain
        if len(recommendation_counts) == 0:
            print("No valid recommendations found")
            return None
        else:
            # Create pie chart
            plt.figure(figsize=(3.64, 2.5))  # Create a new figure
            # Define a function for autopct to show absolute values
            def show_absolute(val):
                absolute = int(round(val/100.*recommendation_counts.sum()))
                return absolute
            
            plt.pie(recommendation_counts, labels=labels, colors=colors, autopct=show_absolute,
                    textprops={'fontsize': 7.5})
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
            # Add text for the total count, outside the pie
            total_recommendations = recommendation_counts.sum()
            plt.text(1.25, -1.0, f"Total Analyst\nRecommendations: {total_recommendations}",
                     horizontalalignment='center', verticalalignment='center', fontsize=5)

        plt.title(title, fontsize=12)
        
        plt.tight_layout()

        filename = os.path.join(output_dir, f"{ticker}_{title.replace(' ', '_').replace("'", "")}_{timestamp}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')


        print(f"Plot saved to {filename}")

        return filename

    except (AttributeError, IndexError, ValueError) as e:
        print(f"Error generating analyst recommendations plot: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    

def plot_candlestick_chart(ticker: str, output_dir: str, timestamp: str):
    """
    Plots the candlestick chart for a given ticker symbol.
    Saves the plot to the output directory.
    """

    length = 30

    company = yf.Ticker(ticker)
    company_name = company.info.get('displayName') if company.info.get('displayName') else company.info.get('shortName')

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=length)).strftime("%Y-%m-%d")

    historical_data = company.history(start=start_date, end=end_date)

    output_path = os.path.join(output_dir, f"candlestick_{ticker}_{start_date}_{end_date}_{timestamp}.png")


    mystyle=mpf.make_mpf_style(base_mpf_style='default', rc={'figure.titlesize':32, 'font.family':FONT_FAMILY})

    fig, axes = mpf.plot(
        historical_data,
        type='candle',
        style=mystyle,
        mav=(5, 30),
        figratio=(16, 9),
        figscale=2.0,
        title=f'{company_name} share price over the last {length} days',
        ylabel='Price ($)',
        volume=True,
        savefig=output_path,
        # show_nontrading=True,
        tight_layout=True,
        scale_padding=dict(left=0.0, right=0.0, top=5.0, bottom=0.0),
        returnfig=True
    )


    print(f"Plot saved to {output_path}")

    return output_path
    


if __name__ == "__main__":
    TIMESTAMP = "20250305_120000"

    PLOTS_DIR = "/home/j/ai/crewAI/finance/stock_analyser/sandboxing"
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_historical_stock_data("AAPL", PLOTS_DIR, TIMESTAMP)
    plot_competitor_prices(["AAPL", "TSLA", "MSFT", "GOOG", "NVDA", "AMZN", "META", "NFLX", "ORCL", "IBM"], PLOTS_DIR, TIMESTAMP)
    plot_anaylsts_recommendations("AAPL", PLOTS_DIR, TIMESTAMP)
    plot_candlestick_chart("AAPL", PLOTS_DIR, TIMESTAMP)