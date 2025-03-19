from typing import Type, Literal
import pandas as pd
import yfinance as yf
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class YFinanceSwingTradingToolInput(BaseModel):
    """Input schema for YFinanceSwingTradingTool."""

    ticker: str = Field(..., description="Ticker symbol of the stock to analyze.")
    period: Literal['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'] = Field(default='1y', description="Period of the stock to analyze. Valid inputs are 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max")
    num_results: int = Field(default=5, description="Maximum number of results to list in the output.")

class YFinanceSwingTradingTool(BaseTool):
    name: str = "YFinanceSwingTradingTool"
    description: str = (
        "Use this tool to get the swing trading signals for a given stock."
    )
    args_schema: Type[BaseModel] = YFinanceSwingTradingToolInput

    def _run(self, ticker: str, period: str = '1y', num_results: int = 5) -> str:
        """
        Get the swing trading signals for a given stock.

        Args:
            ticker: str = Ticker symbol of the stock to analyze.
            period: str = Period of the stock to analyze. Valid inputs are 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max. Default is 1y.
            num_results: int = Maximum number of results to list in the output. Default is 5.

        Returns:
            str = Swing trading signals for the given stock.
        """
        # Get the stock data
        stock = yf.Ticker(ticker)

        if 'd' in period:
            interval, results = '1h', 'hours'
        elif period != 'max' and 'm' in period or period == '1y':
            interval, results = '1d', 'days'
        elif 'y' in period:
            interval, results = '1wk', 'weeks'
        else:
            interval, results = '1mo', 'months'

        df = stock.history(period=period, interval=interval)

        # Calculate EMAs
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        # Ensure there are no NaN values before comparison
        df.dropna(inplace=True)

        signals = f"Historical signals for {ticker} ({period}):\n"

        # 1. Step 1: Identify the Best Market Conditions
        # Condition: Market above 10-21-50 EMA
        df[['Close', 'EMA_10', 'EMA_21', 'EMA_50']] = df[['Close', 'EMA_10', 'EMA_21', 'EMA_50']].astype(float)

        # Condition: Market above 10-21-50 EMA
        df['Above_EMAs'] = (df['Close'].gt(df['EMA_10'])) & (df['Close'].gt(df['EMA_21'])) & (df['Close'].gt(df['EMA_50']))

        days_under_10_ema = df[df['Close'] < df['EMA_10']].index.to_list()
        days_under_21_ema = df[df['Close'] < df['EMA_21']].index.to_list()
        days_under_50_ema = df[df['Close'] < df['EMA_50']].index.to_list()

        if df['Above_EMAs'].all():
            signals += "All closing prices above 10, 21 and 50 day EMAs\n"
        else:
            signals += "Not all closing prices above 10, 21 and 50 day EMAs\n"

            if days_under_10_ema:
                signals += f"Most recent {results} with closing prices below 10 day EMA (max. {num_results} {results} shown):\n"
                for day in days_under_10_ema[-num_results:]:
                    row = df.loc[day]
                    signals += f"  {day.strftime('%Y-%m-%d')}: Close: {row['Close']:.2f}, EMA_10: {row['EMA_10']:.2f}\n"

            if days_under_21_ema:
                signals += f"Most recent {results} with closing prices below 21 day EMA (max. {num_results} {results} shown):\n"
                for day in days_under_21_ema[-num_results:]:
                    row = df.loc[day]
                    signals += f"  {day.strftime('%Y-%m-%d')}: Close: {row['Close']:.2f}, EMA_21: {row['EMA_21']:.2f}\n"

            if days_under_50_ema:
                signals += f"Most recent {results} with closing prices below 50 day EMA (max. {num_results} {results} shown):\n"
                for day in days_under_50_ema[-num_results:]:
                    row = df.loc[day]
                    signals += f"  {day.strftime('%Y-%m-%d')}: Close: {row['Close']:.2f}, EMA_50: {row['EMA_50']:.2f}\n"


        # 2. Base Quality Analysis
        # Detect Volume Dry-up
        # Define a threshold for volume dry-up (e.g., lowest volume in 30 days)
        df['Low_Volume'] = df['Volume'] < df['Volume'].rolling(30).min()
        
        # Filter days where volume is extremely low
        low_volume_days = df[df['Low_Volume']]

        # Format the output for low volume days
        signals += f"Most recent {results} with low volume (max. {num_results} {results} shown):\n"
        if not low_volume_days.empty:
            for index, row in low_volume_days.tail(num_results).iterrows():
                signals += (
                    f"  Date: {index.strftime('%Y-%m-%d')}, "
                    f"Open: ${row['Open']:.2f}, Close: ${row['Close']:.2f}, "
                    f"Volume: {row['Volume']:,}, "
                    f"EMA 10: ${row['EMA_10']:.2f}, EMA 21: ${row['EMA_21']:.2f}, EMA 50: ${row['EMA_50']:.2f}\n"
                )
        else:
            signals += "  No low volume days detected.\n"
        

        # 3. Heavy Selling Days
        # Define heavy selling as a red candle with high volume
        df['Heavy_Sell'] = (df['Close'] < df['Open']) & (df['Volume'] > df['Volume'].rolling(20).mean())

        # Check if the base has too many heavy selling days
        heavy_selling_days = df[df['Heavy_Sell']]

        # Format the output for heavy selling days
        signals += f"Most recent {results} with heavy selling (max. {num_results} {results} shown):\n"
        if not heavy_selling_days.empty:
            for index, row in heavy_selling_days.tail(num_results).iterrows():
                signals += (
                    f"  Date: {index.strftime('%Y-%m-%d')}, "
                    f"Open: ${row['Open']:.2f}, Close: ${row['Close']:.2f}, "
                    f"Volume: {row['Volume']:,}, "
                    f"EMA 10: ${row['EMA_10']:.2f}, EMA 21: ${row['EMA_21']:.2f}, EMA 50: ${row['EMA_50']:.2f}\n"
                )
        else:
            signals += "  No heavy selling days detected.\n"


        # 4. NRIB (Narrow Range Inside Bars)
        df['NRIB'] = (df['High'] - df['Low']) < (df['High'].rolling(5).max() - df['Low'].rolling(5).min()) * 0.3

        # Filter stocks showing NRIB before breakout
        narrow_range_days = df[df['NRIB']]

        # Format the output for narrow range days
        signals += f"Most recent {results} with narrow range inside bars (NRIB) (max. {num_results} {results} shown):\n"
        if not narrow_range_days.empty:
            for index, row in narrow_range_days.tail(num_results).iterrows():
                signals += (
                    f"  Date: {index.strftime('%Y-%m-%d')}, "
                    f"Open: ${row['Open']:.2f}, Close: ${row['Close']:.2f}, "
                    f"Volume: {row['Volume']:,}, "
                    f"EMA 10: ${row['EMA_10']:.2f}, EMA 21: ${row['EMA_21']:.2f}, EMA 50: ${row['EMA_50']:.2f}\n"
                )
        else:
            signals += "  No narrow range days detected.\n"
        

        # # 5. 5–6% Stop Loss (SL)
        # df['Stop_Loss'] = df['Close'] * 0.94  # Setting 6% below the close price

        # # Format the output for stop loss
        # signals += f"{results.capitalize()} with stop loss (max. {num_results} {results} shown):\n"
        # if not df[['Close', 'Stop_Loss']].empty:
        #     for index, row in df[['Close', 'Stop_Loss']].tail(num_results).iterrows():
        #         signals += (
        #             f"  Date: {index.strftime('%Y-%m-%d')}, "
        #             f"Close: ${row['Close']:.2f}, "
        #             f"Stop Loss 6% below close: ${row['Stop_Loss']:.2f}\n\n"
        #         )
        # else:
        #     signals += "  No stop loss data available.\n\n"
        

        # 6. A Prior Upmove
        # Check if stock had a strong prior move before base formation
        df['Prior_Upmove'] = df['Close'] > df['Close'].shift(20) * 1.2  # 20% gain in last 20 days
        prior_upmove_days = df[df['Prior_Upmove']]

        # Format the output for prior upmove days
        signals += f"Most recent {results} with prior upmove before base formation (max. {num_results} {results} shown):\n"
        if not prior_upmove_days.empty:
            for index, row in prior_upmove_days.tail(num_results).iterrows():
                signals += (
                    f"  Date: {index.strftime('%Y-%m-%d')}, "
                    f"Open: ${row['Open']:.2f}, Close: ${row['Close']:.2f}, "
                    f"Volume: {row['Volume']:,}, "
                    f"EMA 10: ${row['EMA_10']:.2f}, EMA 21: ${row['EMA_21']:.2f}, EMA 50: ${row['EMA_50']:.2f}\n"
                )
        else:
            signals += "  No prior upmove days detected.\n"
        
        # signals += f"Prior upmove days: {prior_upmove_days.tail(10).to_dict(orient='index')}\n"


        # 7. A Catalyst or Theme

        sector_key = stock.info.get('sectorKey', 'N/A')
        sector_name = stock.info.get('sectorDisp', 'N/A')

        df_sector = yf.Sector(sector_key).ticker.history(period=period, interval=interval)
        df_sector['Returns'] = df_sector['Close'].pct_change()

        signals += f"{sector_name} sector {period} return: {df_sector['Returns'].sum()*100:.2f}%\n"


        return signals
    

if __name__ == "__main__":
    tool = YFinanceSwingTradingTool()
    print(tool.run("AAPL"))


        
        


        
        

    
