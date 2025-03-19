import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import talib as ta
from stock_analyser.utils.tickers import tickers
from pprint import pprint


def get_stock_data_for_single_ticker(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
	
    df = stock.history(start=start_date, end=end_date, auto_adjust=False)

    smas = [50, 150, 200]
    for sma in smas:
        df["SMA_"+str(sma)]=round(df.iloc[:,4].rolling(window=sma).mean(),2)
		
    # relative_strength
    df['rsi'] = ta.RSI(df['Close'], timeperiod=14)
    upperBB, middleBB, lowerBB = ta.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['upperBB'] = upperBB
    df['middleBB'] = middleBB
    df['lowerBB'] = lowerBB
	
    upperBBrsi, MiddleBBrsi, lowerBBrsi = ta.BBANDS(df['rsi'], timeperiod=50, nbdevup=2, nbdevdn=2, matype=0)
    df['upperBBrsi'] = upperBBrsi
    df['middleBBrsi'] = MiddleBBrsi
    df['lowerBBrsi'] = lowerBBrsi
	
    normrsi = (df['rsi'] - df['lowerBBrsi']) / (df['upperBBrsi'] - df['lowerBBrsi'])
    df['normrsi'] = normrsi
    	
    return df

	
	
def check_conditions_for_single_ticker(ticker: str, start_date: str, end_date: str) -> bool:

    df = get_stock_data_for_single_ticker(ticker, start_date, end_date)
	
    try:
        current_close = df["Adj Close"].iloc[-1]
        moving_average_50 = df["SMA_50"].iloc[-1]
        moving_average_150 = df["SMA_150"].iloc[-1]
        moving_average_200 = df["SMA_200"].iloc[-1]
        low_of_52week = min(df["Adj Close"].iloc[-260:])
        high_of_52week = max(df["Adj Close"].iloc[-260:])
        rs_rating = df['normrsi'].iloc[-1]
        moving_average_200_20 = df["SMA_200"].iloc[-20]


        # Condition 1: Current Price > 150 SMA and > 200 SMA
        cond_1 = current_close > moving_average_150 > moving_average_200
        # Condition 2: 150 SMA and > 200 SMA
        cond_2 = moving_average_150 > moving_average_200
        # Condition 3: 200 SMA trending up for at least 1 month (ideally 4-5 months)
        cond_3 = moving_average_200 > moving_average_200_20
        # Condition 4: 50 SMA > 150 SMA and 50 SMA > 200 SMA
        cond_4 = moving_average_50 > moving_average_150 > moving_average_200
        # Condition 5: Current Price > 50 SMA
        cond_5 = current_close > moving_average_50
        # Condition 6: Current Price is at least 30% above 52 week low
        cond_6 = current_close >= (1.3 * low_of_52week)
        # Condition 7: Current Price is within 25% of 52 week high
        cond_7 = current_close >= (0.75 * high_of_52week)
        # Condition 8: IBD RS rating >70 and the higher the better
        cond_8 = rs_rating > 70

        if all([cond_1, cond_2, cond_3, cond_4, cond_5, cond_6, cond_7, cond_8]):
            screening_results = {
                'Stock': ticker,
                "Normalized Relative Strength Rating": rs_rating,
                "50 Day MA": moving_average_50,
                "150 Day Ma": moving_average_150,
                "200 Day MA": moving_average_200,
                "52 Week Low": low_of_52week,
                "52 week High": high_of_52week
            }
            return screening_results
        else:
            return None
    except Exception as e:
        print(f"No data on {ticker}: {e}")


def get_stock_data_for_multiple_tickers(tickers, start_date, end_date):
    """
    Fetches historical stock data for multiple tickers from Yahoo Finance and calculates technical indicators.

    Args:
        tickers (list): A list of stock ticker symbols (e.g., ['AAPL', 'MSFT']).
        start_date (str): The start date for data retrieval (e.g., '2023-01-01').
        end_date (str): The end date for data retrieval (e.g., '2023-12-31').

    Returns:
        pandas.DataFrame: A DataFrame containing historical stock data and technical indicators,
                          with columns grouped by metric and then individual tickers.  Returns an empty
                          DataFrame if there's an error fetching data.
    """
    try:
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)

        if data.empty:
            print("No data found for the specified tickers and date range.")
            return pd.DataFrame()

        combined_history = data.copy()

        all_ticker_data = []  # List to store DataFrames for each ticker

        for ticker in tickers:
            # Ensure 'Adj Close' exists for the ticker
            if ('Adj Close', ticker) not in combined_history.columns:
                print(f"No 'Adj Close' for {ticker}.")
                continue

            # Create a dictionary to store calculated data for the current ticker
            ticker_data = {}

            # --- Moving Averages ---
            ticker_data[('SMA_50', ticker)] = ta.SMA(combined_history[('Adj Close', ticker)], timeperiod=50)
            ticker_data[('SMA_150', ticker)] = ta.SMA(combined_history[('Adj Close', ticker)], timeperiod=150)
            ticker_data[('SMA_200', ticker)] = ta.SMA(combined_history[('Adj Close', ticker)], timeperiod=200)
            ticker_data[('rsi', ticker)] = ta.RSI(combined_history[('Adj Close', ticker)], timeperiod=14)
            upperBB, middleBB, lowerBB = ta.BBANDS(combined_history[('Adj Close', ticker)], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            ticker_data[('upperBB', ticker)] = upperBB
            ticker_data[('middleBB', ticker)] = middleBB
            ticker_data[('lowerBB', ticker)] = lowerBB
            upperBBrsi, middleBBrsi, lowerBBrsi = ta.BBANDS(combined_history[('rsi', ticker)], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            ticker_data[('upperBBrsi', ticker)] = upperBBrsi
            ticker_data[('middleBBrsi', ticker)] = middleBBrsi
            ticker_data[('lowerBBrsi', ticker)] = lowerBBrsi
            # --- Normalized RSI ---
            # Use .values to get the underlying NumPy array for min/max
            ticker_data[('normrsi', ticker)] = (combined_history[('rsi', ticker)] - combined_history[('rsi', ticker)].min()) / (combined_history[('rsi', ticker)].max() - combined_history[('rsi', ticker)].min())


            # Convert the dictionary to a DataFrame and append
            ticker_df = pd.DataFrame(ticker_data)
            all_ticker_data.append(ticker_df)

        # Concatenate all ticker data at once
        combined_history = pd.concat([combined_history] + all_ticker_data, axis=1)
        return combined_history


    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()
    

def check_conditions_for_multiple_tickers(tickers, start_date, end_date):
    combined_history = get_stock_data_for_multiple_tickers(tickers, start_date, end_date)

    screening_results = []

    for ticker in tickers:
        try:
            # --- Create a temporary DataFrame for the single ticker ---
            # Efficiently select columns for the current ticker using a list comprehension
            temp_df = combined_history[[col for col in combined_history.columns if col[1] == ticker]].copy()
            # Rename columns to remove the MultiIndex
            temp_df.columns = [col[0] for col in temp_df.columns]


            # --- Use the single-ticker function ---
            if not temp_df.empty:  # Check if temp_df has data
                # Add a dummy Ticker column to temp_df
                temp_df['Ticker'] = ticker
                # Call the single-ticker function (modified version)
                single_result = check_conditions_for_single_ticker_from_df(temp_df)

                if single_result:
                    screening_results.append(single_result)

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    return screening_results



def check_conditions_for_single_ticker_from_df(df):
    """
    Checks conditions for a single ticker, taking a DataFrame as input.
    This is a modified version of the original function to work with a pre-fetched DataFrame.
    """
    try:
        ticker = df['Ticker'].iloc[0] # Get the ticker symbol
        current_close = df["Adj Close"].iloc[-1]
        moving_average_50 = df["SMA_50"].iloc[-1]
        moving_average_150 = df["SMA_150"].iloc[-1]
        moving_average_200 = df["SMA_200"].iloc[-1]
        low_of_52week = min(df["Adj Close"].iloc[-260:])
        high_of_52week = max(df["Adj Close"].iloc[-260:])
        rs_rating = df['normrsi'].iloc[-1]
        moving_average_200_20 = df["SMA_200"].iloc[-20]


        # Condition 1: Current Price > 150 SMA and > 200 SMA
        cond_1 = current_close > moving_average_150 > moving_average_200
        # Condition 2: 150 SMA and > 200 SMA
        cond_2 = moving_average_150 > moving_average_200
        # Condition 3: 200 SMA trending up for at least 1 month (ideally 4-5 months)
        cond_3 = moving_average_200 > moving_average_200_20
        # Condition 4: 50 SMA > 150 SMA and 50 SMA > 200 SMA
        cond_4 = moving_average_50 > moving_average_150 > moving_average_200
        # Condition 5: Current Price > 50 SMA
        cond_5 = current_close > moving_average_50
        # Condition 6: Current Price is at least 30% above 52 week low
        cond_6 = current_close >= (1.3 * low_of_52week)
        # Condition 7: Current Price is within 25% of 52 week high
        cond_7 = current_close >= (0.75 * high_of_52week)
        # Condition 8: IBD RS rating >70 and the higher the better
        cond_8 = rs_rating > 70

        if all([cond_1, cond_2, cond_3, cond_4, cond_5, cond_6, cond_7, cond_8]):
            screening_results = {
                'Stock': ticker,
                "Normalized Relative Strength Rating": rs_rating,
                "50 Day MA": moving_average_50,
                "150 Day Ma": moving_average_150,
                "200 Day MA": moving_average_200,
                "52 Week Low": low_of_52week,
                "52 week High": high_of_52week
            }
            return screening_results
        else:
            return None  # Explicitly return None if conditions not met
    except Exception as e:
        print(f"No data on {ticker}: {e}")
        return None # Return None in case of exception


def check_conditions_for_single_ticker(ticker: str, start_date: str, end_date: str) -> bool:
    # --- This function is no longer needed, but we keep it to avoid breaking existing calls ---
    # It now just fetches the data and calls the new function
    df = get_stock_data_for_single_ticker(ticker, start_date, end_date)
    df['Ticker'] = ticker # Add the ticker
    return check_conditions_for_single_ticker_from_df(df)



if __name__ == "__main__":
    ticker_list = ['AAPL', 'NVDA', 'MSFT', 'TSLA', 'GOOG', 'AMZN', 'META', 'TSM', 'NFLX', 'ORCL', 'IBM', 'QCOM', 'WMT', 'JNJ', 'VZ', 'T', 'MMM', 'BA', 'CAT', 'CSCO', 'CVX', 'KO', 'PFE', 'WBA', 'DIS', 'GS', 'JPM', 'MS', 'NKE', 'PG', 'TRMB', 'V', 'WMT', 'XOM', 'ZM', 'ZWS', 'ZYBT', 'ZYME', 'ZYXI']

    screening_results = check_conditions_for_multiple_tickers(ticker_list, "2024-01-01", "2025-03-19")
    pprint(screening_results)


    # for ticker in ticker_list:
    #     print("-"*100)
    #     print(f"Screening {ticker}")
    #     result = check_conditions_for_single_ticker(ticker, "2024-01-01", "2025-03-19")
    #     time.sleep(3)

    #     passed = []
    #     if result:
    #         pprint(result)
    #         print("-"*100)
    #         print("\n\n")
    #         passed.append(ticker)
    #     else:
    #         print(f"Screening {ticker} - Failed", end="\r")

    # print(f"Stocks that passed the screening: {passed}")


    
    





