import yfinance as yf
import pandas as pd
import mplfinance as mpf
from pathlib import Path
from datetime import datetime
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from stock_analyser.utils.constants import FONT_FAMILY


RED = '#fd6b6c'
FAINT_RED = '#de1a24'
FAINT_GREEN = '#3f8f29'
GREEN = '#4dc790'


def get_historical_prices(ticker: str, period: str) -> pd.DataFrame:
    """
    Get historical prices for a given ticker.
    """
    company_ticker = yf.Ticker(ticker)
    historical_prices = company_ticker.history(period=period, rounding=True)
    return historical_prices


def generate_awesome_oscillator_color(df):
    awesome_oscillator_color = []
    awesome_oscillator_color.clear()
    for i in range (0,len(df["AO"])):
        if df["AO"].iloc[i] >= 0 and df["AO"].iloc[i-1] < df["AO"].iloc[i]:
            awesome_oscillator_color.append(GREEN)
        elif df["AO"].iloc[i] >= 0 and df["AO"].iloc[i-1] > df["AO"].iloc[i]:
            awesome_oscillator_color.append(RED)
        elif df["AO"].iloc[i] < 0 and df["AO"].iloc[i-1] > df["AO"].iloc[i] :
            awesome_oscillator_color.append(FAINT_RED)
        elif df["AO"].iloc[i] < 0 and df["AO"].iloc[i-1] < df["AO"].iloc[i] :
            awesome_oscillator_color.append(FAINT_GREEN)
        else:
            awesome_oscillator_color.append('#000000')
    return awesome_oscillator_color


def plot_awesome_oscillator(ticker: str, period: str, output_dir: str, timestamp: str):
    """
    Plot the awesome oscillator for a given ticker.
    """
    try:
        df = get_historical_prices(ticker, period)
    except Exception as e:
        print(f"Error getting historical prices: {e}")
        return None

    # Calculate Median Price
    df['Median Price'] = (df['High'] + df['Low']) / 2

    #Calculation For Simple Moving Average For Length 34 as Long SMA
    df['SMA34'] = df['Median Price'].rolling(window=34).mean()

    #Calculation For Simple Moving Average For Length 5 as Short SMA
    df['SMA5'] = df['Median Price'].rolling(window=5).mean()  # SMA5 on Median Price

    # Calculate other MAVs
    df['SMA25'] = df['Close'].rolling(window=25).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()

    #Calculation For Awesome Oscillator
    df['AO'] = df['SMA5'] - df['SMA34']

    # List of Color Assiging To Awesome Oscillator
    awesome_oscillator_color = generate_awesome_oscillator_color(df)

    # Data Extracted And New Variable Applied
    awesome_oscillator = df[['AO']]

    # Plotting Awesome Oscillator
    ao = [
        mpf.make_addplot(
            awesome_oscillator,
            type='bar',
            width=1.0,
            color=awesome_oscillator_color,
            panel=1,
            alpha=1,
            secondary_y=False,
            y_on_right=False,
            ylabel="AO Value"
        )
    ]

    # Plot MAVs, collect lines for legend.  Only plot SMA on Close.
    mav_plots = [
      mpf.make_addplot(df['SMA25'], color='orange', width=1.0),
      mpf.make_addplot(df['SMA50'], color='purple', width=1.0),
      mpf.make_addplot(df['SMA200'], color='brown', width=1.0),
    ]
    ao.extend(mav_plots)


    mystyle=mpf.make_mpf_style(base_mpf_style='yahoo',rc={'axes.labelsize':11})

    # Plotting Awesome Oscillator and saving
    plot_dir = output_dir
    file_path = f'{plot_dir}/{ticker}_{period}_awesome_oscillator_{timestamp}.png'

    # Use returnfig=True to get the figure and axes objects
    fig, axes = mpf.plot(df,
        type='hollow_and_filled',
        mav=(5, 25, 50, 200),  # Keep mav here for correct x-axis scaling
        volume=True,
        volume_panel=2,
        figsize=(16, 9),
        figscale=1.1,
        style=mystyle,
        title=f'{ticker} {period} Price History',
        ylabel='Price',
        addplot=ao,
        tight_layout=True,
        scale_padding=dict(left=0.5, right=2.0, top=1.0, bottom=1.0),
        show_nontrading=True,
        returnfig=True,
    )

    # Create legend patches and labels.  Use the *new* colors.
    red_negative_patch = mpatches.Patch(color=RED, label='AO Negative, Decreasing')
    green_negative_patch = mpatches.Patch(color=FAINT_GREEN, label='AO Negative, Increasing')
    red_positive_patch = mpatches.Patch(color=FAINT_RED, label='AO Positive, Decreasing')
    green_positive_patch = mpatches.Patch(color=GREEN, label='AO Positive, Increasing')

    # Create legend lines for MAVs
    mav25_line = mlines.Line2D([], [], color='orange', label='SMA 25')
    mav50_line = mlines.Line2D([], [], color='purple', label='SMA 50')
    mav200_line = mlines.Line2D([], [], color='brown', label='SMA 200')

    # Add the legend to the *second* axes object (the AO subplot)
    # Combine AO and MAV legends
    axes[0].legend(
        handles=[
            red_negative_patch,
            green_negative_patch,
            red_positive_patch,
            green_positive_patch,
            mav25_line,
            mav50_line,
            mav200_line
        ],
        loc='upper left'
    )


    # Save the figure *after* adding the legend and text
    try:
        fig.savefig(str(file_path))
    except Exception as e:
        print(f"Error saving plot: {e}")
        return None

    print(f"Plot saved to {file_path}")

    return file_path


if __name__ == "__main__":
    ticker = "MSFT"
    period = "1y"
    plots_dir = "plots/ao"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_awesome_oscillator(ticker, period, plots_dir, timestamp)
