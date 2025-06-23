import yfinance as yf
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions (Integrated or Kept Separate) ---

def safe_get_row(df: pd.DataFrame, row_name: str, alternative_names: Optional[List[str]] = None) -> Optional[pd.Series]:
    """
    Safely get a row from a DataFrame, handling KeyError and empty DataFrames.

    Args:
        df (pd.DataFrame): The DataFrame to get the row from.
        row_name (str): The primary name of the row to get.
        alternative_names (list, optional): Alternative names to try if row_name is not found.

    Returns:
        pd.Series or None: The row data or None if not found or if df is invalid.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        # logger.debug(f"Invalid or empty DataFrame passed to safe_get_row for row '{row_name}'.")
        return None

    try:
        # Check if the primary row name exists
        if row_name in df.index:
            return df.loc[row_name]
    except KeyError: # This might be redundant given the 'in' check, but belt-and-suspenders
        pass # Continue to check alternative names

    # Check alternative names if provided
    if alternative_names:
        for alt_name in alternative_names:
            try:
                 if alt_name in df.index:
                    # logger.debug(f"Found row using alternative name '{alt_name}' for primary '{row_name}'.")
                    return df.loc[alt_name]
            except KeyError:
                continue # Try the next alternative name

    # logger.warning(f"Row '{row_name}' (and alternatives: {alternative_names}) not found in DataFrame.")
    return None


def filter_valid_values(series: Optional[pd.Series]) -> List[Any]:
    """
    Filter a pandas Series to only include valid numeric values (not None or NaN).

    Args:
        series: A pandas Series or None.

    Returns:
        list: A list of valid numeric values found in the series.
    """
    if series is None or not isinstance(series, pd.Series):
        return []

    # Filter out None and NaN values
    valid_vals = series.dropna()

    # Further ensure values are numeric if possible (though dropna often handles this)
    numeric_vals = [val for val in valid_vals if isinstance(val, (int, float, np.number)) and not np.isnan(val)]

    return numeric_vals

# --- Service Class ---

class YFinanceDataService:
    """
    A centralized service for fetching, caching, and processing data from yfinance.
    Handles ticker instantiation, data retrieval, currency conversion, and basic error handling.
    """
    def __init__(self):
        """Initializes caches for ticker objects, fetched data, and exchange rates."""
        self.ticker_cache: Dict[str, yf.Ticker] = {}
        self.data_cache: Dict[Tuple[str, str], Any] = {} # Key: (ticker, data_type), Value: data
        self.history_cache: Dict[Tuple[str, str, str], pd.DataFrame] = {} # Key: (ticker, period, interval)
        self.exchange_rate_cache: Dict[str, float] = {'USD': 1.0} # Cache for currency conversion rates to USD

    # --- Private Helper Methods ---

    def _get_ticker_object(self, ticker: str) -> Optional[yf.Ticker]:
        """
        Retrieves or creates a yfinance.Ticker object, using caching.

        Args:
            ticker (str): The stock ticker symbol.

        Returns:
            yf.Ticker or None: The Ticker object or None if instantiation fails.
        """
        ticker_upper = ticker.upper()
        if ticker_upper in self.ticker_cache:
            return self.ticker_cache[ticker_upper]
        try:
            ticker_obj = yf.Ticker(ticker_upper)
            # Basic validation: Check if info can be fetched (even if empty)
            _ = ticker_obj.info
            self.ticker_cache[ticker_upper] = ticker_obj
            # logger.debug(f"Created and cached yf.Ticker object for {ticker_upper}")
            return ticker_obj
        except Exception as e:
            logger.error(f"Failed to instantiate yf.Ticker for {ticker_upper}: {e}")
            return None

    @lru_cache(maxsize=50) # Cache results for common currency pairs
    def _get_exchange_rate_to_usd(self, source_currency: str) -> float:
        """
        Fetches and caches the exchange rate for converting source_currency to USD.

        Args:
            source_currency (str): The source currency code (e.g., 'EUR', 'JPY').

        Returns:
            float: The exchange rate to multiply by to get USD. Returns 1.0 if conversion fails or source is USD.
        """
        source_currency = source_currency.upper()
        if source_currency == 'USD':
            return 1.0
        if source_currency in self.exchange_rate_cache:
            return self.exchange_rate_cache[source_currency]

        try:
            rate_ticker_symbol = f"{source_currency}USD=X"
            rate_ticker = yf.Ticker(rate_ticker_symbol)
            # Fetch only the most recent 'Close' price
            history = rate_ticker.history(period='1d', interval='1d')
            if not history.empty and 'Close' in history.columns:
                exchange_rate = history['Close'].iloc[-1]
                if exchange_rate > 0:
                    self.exchange_rate_cache[source_currency] = exchange_rate
                    logger.debug(f"Fetched and cached exchange rate for {source_currency} to USD: {exchange_rate}")
                    return exchange_rate
            logger.warning(f"Could not fetch valid exchange rate for {source_currency} to USD.")
            return 1.0 # Fallback
        except Exception as e:
            logger.error(f"Error fetching exchange rate for {source_currency} to USD: {e}")
            return 1.0 # Fallback

    def _fetch_and_cache(self, ticker: str, data_type: str, fetch_func: callable) -> Any:
        """
        Generic fetch and cache mechanism for dictionary-like data (e.g., info).

        Args:
            ticker (str): Ticker symbol.
            data_type (str): Identifier for the data type (e.g., 'info', 'sustainability').
            fetch_func (callable): The function from yf.Ticker object to call (e.g., ticker_obj.info).

        Returns:
            Any: The fetched data (usually dict) or None on error.
        """
        cache_key = (ticker.upper(), data_type)
        if cache_key in self.data_cache:
            # logger.debug(f"Cache hit for {data_type} of {ticker.upper()}")
            return self.data_cache[cache_key]

        ticker_obj = self._get_ticker_object(ticker)
        if not ticker_obj:
            return None # Error handled in _get_ticker_object

        try:
            data = fetch_func() # Call the specific fetch function (e.g., ticker_obj.info)
            if data is None: # Check for None explicitly
                 logger.warning(f"Fetching {data_type} for {ticker.upper()} returned None.")
                 data = {} # Return empty dict for consistency
            elif isinstance(data, pd.DataFrame) and data.empty:
                 logger.warning(f"Fetching {data_type} for {ticker.upper()} returned an empty DataFrame.")
                 data = pd.DataFrame() # Return empty DataFrame
            elif isinstance(data, dict) and not data:
                 logger.warning(f"Fetching {data_type} for {ticker.upper()} returned an empty dictionary.")
                 # Keep data as empty dict

            self.data_cache[cache_key] = data
            # logger.debug(f"Fetched and cached {data_type} for {ticker.upper()}")
            return data
        except AttributeError:
             logger.error(f"yf.Ticker object for {ticker.upper()} does not have attribute for '{data_type}'.")
             return None
        except Exception as e:
            logger.error(f"Error fetching {data_type} for {ticker.upper()}: {e}")
            self.data_cache[cache_key] = None # Cache the failure to prevent retries
            return None

    def _fetch_financial_statement(self, ticker: str, statement_type: str, fetch_func: callable) -> pd.DataFrame:
        """
        Fetches, caches, and converts currency for financial statements (DataFrame).

        Args:
            ticker (str): Ticker symbol.
            statement_type (str): 'financials', 'balance_sheet', 'cashflow'.
            fetch_func (callable): The function from yf.Ticker (e.g., ticker_obj.financials).

        Returns:
            pd.DataFrame: Processed financial statement (converted to USD) or empty DataFrame on error.
        """
        cache_key = (ticker.upper(), statement_type)
        if cache_key in self.data_cache:
            # logger.debug(f"Cache hit for {statement_type} of {ticker.upper()}")
            return self.data_cache[cache_key]

        ticker_obj = self._get_ticker_object(ticker)
        if not ticker_obj:
            return pd.DataFrame()

        try:
            df = fetch_func()
            if df is None or df.empty:
                logger.warning(f"Fetching {statement_type} for {ticker.upper()} returned empty data.")
                self.data_cache[cache_key] = pd.DataFrame()
                return pd.DataFrame()

            # Get currency and convert
            info_data = self.get_info(ticker) # Use service method to get cached info
            source_currency = info_data.get('financialCurrency', 'USD') if info_data else 'USD'

            exchange_rate = self._get_exchange_rate_to_usd(source_currency)

            if exchange_rate != 1.0:
                # Identify numeric columns for conversion, handle potential errors
                numeric_cols = df.select_dtypes(include=np.number).columns
                try:
                    # Apply conversion, fill NaN after potential multiplication issues
                    df[numeric_cols] = df[numeric_cols].apply(lambda x: pd.to_numeric(x * exchange_rate, errors='coerce')).fillna(df)
                    logger.debug(f"Applied currency conversion (rate: {exchange_rate}) to {statement_type} for {ticker.upper()}")
                except Exception as conversion_error:
                     logger.error(f"Error during currency conversion for {statement_type} of {ticker.upper()}: {conversion_error}")
                     # Return original df if conversion fails? Or empty? Let's return original for now.
            else:
                logger.debug(f"No currency conversion needed for {statement_type} of {ticker.upper()} (already USD or rate=1.0).")

            self.data_cache[cache_key] = df
            # logger.debug(f"Fetched and cached {statement_type} for {ticker.upper()}")
            return df
        except AttributeError:
            logger.error(f"yf.Ticker object for {ticker.upper()} does not have attribute for '{statement_type}'.")
            self.data_cache[cache_key] = pd.DataFrame()
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching {statement_type} for {ticker.upper()}: {e}")
            self.data_cache[cache_key] = pd.DataFrame()
            return pd.DataFrame()

    # --- Public Data Fetching Methods ---

    def get_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetches and caches the .info dictionary for a ticker."""
        ticker_obj = self._get_ticker_object(ticker)
        return self._fetch_and_cache(ticker, 'info', lambda: getattr(ticker_obj, 'info', None))

    def get_financials(self, ticker: str) -> pd.DataFrame:
        """Fetches, caches, and currency-converts the .financials DataFrame."""
        ticker_obj = self._get_ticker_object(ticker)
        return self._fetch_financial_statement(ticker, 'financials', lambda: getattr(ticker_obj, 'financials', None))

    def get_balance_sheet(self, ticker: str) -> pd.DataFrame:
        """Fetches, caches, and currency-converts the .balance_sheet DataFrame."""
        ticker_obj = self._get_ticker_object(ticker)
        return self._fetch_financial_statement(ticker, 'balance_sheet', lambda: getattr(ticker_obj, 'balance_sheet', None))

    def get_cashflow(self, ticker: str) -> pd.DataFrame:
        """Fetches, caches, and currency-converts the .cashflow DataFrame."""
        ticker_obj = self._get_ticker_object(ticker)
        return self._fetch_financial_statement(ticker, 'cashflow', lambda: getattr(ticker_obj, 'cashflow', None))

    def get_income_stmt(self, ticker: str) -> pd.DataFrame:
        """Fetches, caches, and currency-converts the .income_stmt DataFrame."""
        ticker_obj = self._get_ticker_object(ticker)
        return self._fetch_financial_statement(ticker, 'income_stmt', lambda: getattr(ticker_obj, 'income_stmt', None))


    def get_history(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Fetches and caches the .history DataFrame for a ticker, period, and interval."""
        cache_key = (ticker.upper(), period, interval)
        if cache_key in self.history_cache:
            # logger.debug(f"Cache hit for history of {ticker.upper()} ({period}, {interval})")
            return self.history_cache[cache_key]

        ticker_obj = self._get_ticker_object(ticker)
        if not ticker_obj:
            return pd.DataFrame()

        try:
            history_df = ticker_obj.history(period=period, interval=interval, auto_adjust=False)
            if history_df.empty:
                logger.warning(f"Fetching history for {ticker.upper()} ({period}, {interval}) returned empty data.")
            self.history_cache[cache_key] = history_df
            # logger.debug(f"Fetched and cached history for {ticker.upper()} ({period}, {interval})")
            return history_df
        except Exception as e:
            logger.error(f"Error fetching history for {ticker.upper()} ({period}, {interval}): {e}")
            self.history_cache[cache_key] = pd.DataFrame() # Cache failure
            return pd.DataFrame()

    def get_recommendations(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetches and caches the .recommendations DataFrame."""
        ticker_obj = self._get_ticker_object(ticker)
        # Recommendations often don't need currency conversion
        return self._fetch_and_cache(ticker, 'recommendations', lambda: getattr(ticker_obj, 'recommendations', None))

    def get_sustainability(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetches and caches the .sustainability DataFrame."""
        ticker_obj = self._get_ticker_object(ticker)
        return self._fetch_and_cache(ticker, 'sustainability', lambda: getattr(ticker_obj, 'sustainability', None))

    def get_insider_purchases(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetches and caches the .insider_purchases DataFrame."""
        ticker_obj = self._get_ticker_object(ticker)
        return self._fetch_and_cache(ticker, 'insider_purchases', lambda: getattr(ticker_obj, 'insider_purchases', None))

    def get_institutional_holders(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetches and caches the .institutional_holders DataFrame."""
        ticker_obj = self._get_ticker_object(ticker)
        return self._fetch_and_cache(ticker, 'institutional_holders', lambda: getattr(ticker_obj, 'institutional_holders', None))

    def get_major_holders(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetches and caches the .major_holders DataFrame."""
        ticker_obj = self._get_ticker_object(ticker)
        return self._fetch_and_cache(ticker, 'major_holders', lambda: getattr(ticker_obj, 'major_holders', None))

    # Add other methods as needed following the same pattern...
    # e.g., get_earnings_dates, get_dividends, get_splits, etc.

# --- Example Usage ---
if __name__ == "__main__":
    service = YFinanceDataService()

    tickers_to_test = ["AAPL", "MSFT", "HMC", "NONEXISTENT"] # HMC is Honda (JPY)

    for ticker in tickers_to_test:
        print(f"\n--- Testing Ticker: {ticker} ---")

        # Test get_info
        print("\nFetching Info...")
        info = service.get_info(ticker)
        if info:
            print(f"  Info fetched. Currency: {info.get('currency')}, Market Cap: {info.get('marketCap')}")
            # Fetch again to test cache
            info_cached = service.get_info(ticker)
            print(f"  Info fetched from cache: {'Yes' if info is info_cached else 'No'}")
        else:
            print("  Failed to fetch info.")

        # Test get_financials
        print("\nFetching Financials (converted to USD)...")
        financials = service.get_financials(ticker)
        if not financials.empty:
            print(f"  Financials fetched (shape: {financials.shape}). First few rows/cols:")
            print(financials.iloc[:3, :2])
             # Fetch again to test cache
            financials_cached = service.get_financials(ticker)
            print(f"  Financials fetched from cache: {'Yes' if financials is financials_cached else 'No'}")
        else:
            print("  Failed to fetch financials or data is empty.")

        # Test get_balance_sheet
        print("\nFetching Balance Sheet (converted to USD)...")
        balance_sheet = service.get_balance_sheet(ticker)
        if not balance_sheet.empty:
            print(f"  Balance Sheet fetched (shape: {balance_sheet.shape}).")
        else:
            print("  Failed to fetch balance sheet or data is empty.")

        # Test get_cashflow
        print("\nFetching Cash Flow (converted to USD)...")
        cashflow = service.get_cashflow(ticker)
        if not cashflow.empty:
            print(f"  Cash Flow fetched (shape: {cashflow.shape}).")
        else:
            print("  Failed to fetch cash flow or data is empty.")

        # Test get_history
        print("\nFetching History (1y, 1d)...")
        history = service.get_history(ticker, period="1y", interval="1d")
        if not history.empty:
            print(f"  History fetched (shape: {history.shape}). Last close: {history['Close'].iloc[-1]:.2f}")
            # Fetch again to test cache
            history_cached = service.get_history(ticker, period="1y", interval="1d")
            print(f"  History fetched from cache: {'Yes' if history is history_cached else 'No'}")
        else:
            print("  Failed to fetch history or data is empty.")

        # Test get_recommendations
        print("\nFetching Recommendations...")
        recs = service.get_recommendations(ticker)
        if recs is not None and not recs.empty:
             print(f"  Recommendations fetched (shape: {recs.shape}). Most recent:")
             print(recs.tail(1))
        else:
             print("  No recommendations data available or fetch failed.")

        # Test get_sustainability
        print("\nFetching Sustainability...")
        sust = service.get_sustainability(ticker)
        if sust is not None and not sust.empty:
             print(f"  Sustainability fetched (shape: {sust.shape}). Total ESG Score:")
             print(sust.loc['totalEsg'])
        elif sust is not None and sust.empty:
             print("  Sustainability data is available but empty.")
        else:
             print("  No sustainability data available or fetch failed.")

        print(f"--- End Testing {ticker} ---")

    # Demonstrate safe_get_row
    print("\n--- Testing safe_get_row ---")
    sample_df = service.get_financials("AAPL") # Get a sample df
    if not sample_df.empty:
        print("Sample Financials DataFrame (first few rows/cols):")
        print(sample_df.iloc[:5,:2])
        net_income_row = safe_get_row(sample_df, "Net Income")
        if net_income_row is not None:
            print(f"\nNet Income Row:\n{net_income_row[:2]}") # Print first 2 values
        else:
            print("\nNet Income row not found.")

        ebit_row = safe_get_row(sample_df, "EBIT", ["Operating Income"]) # Example with alternative
        if ebit_row is not None:
            print(f"\nEBIT Row (using primary or alternative 'Operating Income'):\n{ebit_row[:2]}")
        else:
            print("\nEBIT row (or alternative) not found.")

        nonexistent_row = safe_get_row(sample_df, "NonExistentRow")
        if nonexistent_row is None:
            print("\nCorrectly handled non-existent row.")
        else:
            print("\nError: Should not have found non-existent row.")
    else:
        print("\nCould not get sample DataFrame for safe_get_row test.")