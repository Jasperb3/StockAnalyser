import pandas as pd
import yfinance as yf
import mplfinance as mpf
from stock_analyser.utils.constants import FONT_FAMILY


# Add MACD as subplot
def MACD(df, window_slow, window_fast, window_signal):
    macd = pd.DataFrame()
    macd['ema_slow'] = df['Close'].ewm(span=window_slow).mean()
    macd['ema_fast'] = df['Close'].ewm(span=window_fast).mean()
    macd['macd'] = macd['ema_slow'] - macd['ema_fast']
    macd['signal'] = macd['macd'].ewm(span=window_signal).mean()
    macd['diff'] = macd['macd'] - macd['signal']
    macd['bar_positive'] = macd['diff'].map(lambda x: x if x > 0 else 0)
    macd['bar_negative'] = macd['diff'].map(lambda x: x if x < 0 else 0)
    return macd


def Stochastic(df, window, smooth_window):
    stochastic = pd.DataFrame()
    stochastic['%K'] = ((df['Close'] - df['Low'].rolling(window).min()) \
                        / (df['High'].rolling(window).max() - df['Low'].rolling(window).min())) * 100
    stochastic['%D'] = stochastic['%K'].rolling(smooth_window).mean()
    stochastic['%SD'] = stochastic['%D'].rolling(smooth_window).mean()
    stochastic['UL'] = 80
    stochastic['DL'] = 20
    return stochastic


def plot_macd_stochastic(ticker: str, period: str, output_dir: str, timestamp: str):
    company_ticker = yf.Ticker(ticker)
    company_name = company_ticker.info.get('displayName') if company_ticker.info.get('displayName') else company_ticker.info.get('shortName')
    df = company_ticker.history(period=period)
    macd = MACD(df, 12, 26, 9)
    stochastic = Stochastic(df, 14, 3)


    plots  = [
        mpf.make_addplot((macd['macd']), color='#606060', panel=2, ylabel='MACD (12,26,9)', secondary_y=False),
        mpf.make_addplot((macd['signal']), color='#1f77b4', panel=2, secondary_y=False),
        mpf.make_addplot((macd['bar_positive']), type='bar', color='#4dc790', panel=2),
        mpf.make_addplot((macd['bar_negative']), type='bar', color='#fd6b6c', panel=2),
        mpf.make_addplot((stochastic[['%D', '%SD', 'UL', 'DL']]),
                        ylim=[0, 100], panel=3, ylabel='Stoch (14,3)')
    ]

    fig, axes = mpf.plot(
        df,
        type='candle',
        style='yahoo',
        mav=(5,20),
        volume=True,
        addplot=plots,
        panel_ratios=(3,1,3,3),
        figsize=(16, 9),
        figscale=1.1,
        tight_layout=True,
        scale_padding=dict(left=0.25, right=2.0, top=1.75, bottom=1.0),
        returnfig=True,
    )

    # Add title
    axes[0].set_title(f'{company_name} {period} MACD and Stochastic', fontsize=24, fontfamily=FONT_FAMILY)


    file_path = f'{output_dir}/{ticker}_{period}_macd_stochastic_ao_{timestamp}.png'
    
    try:
        fig.savefig(file_path, dpi=300)
    except Exception as e:
        print(f"Error saving plot: {e}")
        return None

    print(f"Plot saved to {file_path}")

    return file_path


if __name__ == "__main__":
    ticker = 'HMC'
    period = '6mo'
    output_dir = 'plots/macd_stochastic'
    timestamp = '20250310_094855'
    plot_macd_stochastic(ticker, period, output_dir, timestamp)
