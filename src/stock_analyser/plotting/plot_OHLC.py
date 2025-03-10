import mplfinance as mpf
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.lines as mlines


def plot_OHLC(ticker: str):
    """
    Plots the OHLC (Open, High, Low, Close) chart for a given ticker symbol for a given range of dates.
    Saves the plot to the output directory.
    """

    company = yf.Ticker(ticker)

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=35)).strftime("%Y-%m-%d")

    historical_data = company.history(start=start_date, end=end_date)

    output_path = f"/home/j/ai/crewAI/finance/stock_analyser/sandboxing/OHLC_{ticker}_{start_date}_{end_date}.png"

    mystyle=mpf.make_mpf_style(rc={'figure.titlesize': 48})

    fig, axes = mpf.plot(
        historical_data,
        type='candle',
        style=mystyle,
        mav=(5, 30),
        figratio=(16, 9),
        figscale=2.0,
        title=f'{ticker} OHLC 30 days',
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
    plot_OHLC("AAPL")


