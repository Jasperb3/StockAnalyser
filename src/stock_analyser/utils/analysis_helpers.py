"""
Shared utility functions for investor analysis modules.

This module provides common helper functions used across all investor-style
analysis implementations (Graham, Buffett, Munger, Wood, Druckenmiller, Ackman).
"""

from typing import Optional, List, Union
import yfinance as yf
import numpy as np
import pandas as pd


def get_ticker(ticker: str) -> yf.Ticker:
    """
    Get a yfinance Ticker object for the given ticker symbol.

    Args:
        ticker: The stock ticker symbol

    Returns:
        A yfinance Ticker object

    Raises:
        Exception: If ticker data cannot be retrieved
    """
    try:
        return yf.Ticker(ticker.upper())
    except Exception as e:
        raise Exception(f"Failed to get ticker data for {ticker}: {str(e)}")


def safe_get_row(
    df: pd.DataFrame,
    row_name: str,
    alternative_names: Optional[List[str]] = None
) -> Optional[pd.Series]:
    """
    Safely get a row from a DataFrame, handling KeyError and empty DataFrames.

    Args:
        df: The DataFrame to get the row from
        row_name: The name of the row to get
        alternative_names: Alternative names to try if row_name is not found

    Returns:
        The row data or None if not found
    """
    if df is None or df.empty:
        return None

    try:
        return df.loc[row_name]
    except KeyError:
        if alternative_names:
            for alt_name in alternative_names:
                try:
                    return df.loc[alt_name]
                except KeyError:
                    continue
        return None


def filter_valid_values(series: Optional[Union[pd.Series, list]]) -> List[float]:
    """
    Filter a series to only include valid numeric values (not None or NaN).

    Args:
        series: A pandas Series or list-like object

    Returns:
        A list of valid numeric values
    """
    if series is None:
        return []

    if isinstance(series, pd.Series):
        return [val for val in series if val is not None and not np.isnan(val)]
    else:
        return [val for val in series if val is not None and not np.isnan(val)]


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0
) -> float:
    """
    Safely divide two numbers, returning a default value if division by zero.

    Args:
        numerator: The numerator
        denominator: The denominator
        default: Value to return if denominator is zero or invalid

    Returns:
        The result of division or the default value
    """
    if denominator is None or np.isnan(denominator) or denominator == 0:
        return default
    if numerator is None or np.isnan(numerator):
        return default
    return numerator / denominator


def validate_dataframe_value(
    df: pd.DataFrame,
    row_name: str,
    index: int = 0,
    alternative_names: Optional[List[str]] = None
) -> Optional[float]:
    """
    Safely extract a value from a DataFrame with validation.

    Args:
        df: The DataFrame to extract from
        row_name: The row name to look for
        index: The column index to extract (default 0 for most recent)
        alternative_names: Alternative row names to try

    Returns:
        The extracted value or None if not available
    """
    row = safe_get_row(df, row_name, alternative_names)
    if row is None or row.empty:
        return None

    valid_values = filter_valid_values(row)
    if not valid_values or len(valid_values) <= index:
        return None

    return valid_values[index]


def calculate_growth_rate(
    values: List[float],
    latest_first: bool = True
) -> Optional[float]:
    """
    Calculate growth rate from earliest to latest value in a list.

    Args:
        values: List of values (assumes reverse chronological if latest_first=True)
        latest_first: Whether the list is ordered newest-first (default True for yfinance)

    Returns:
        Growth rate as decimal (e.g., 0.15 for 15% growth) or None if calculation fails
    """
    if not values or len(values) < 2:
        return None

    latest = values[0] if latest_first else values[-1]

    # Find valid earliest value (non-zero)
    earliest = None
    search_range = range(len(values) - 1, -1, -1) if latest_first else range(len(values))
    for i in search_range:
        if values[i] != 0:
            earliest = values[i]
            break

    if earliest is None or earliest == 0:
        return None

    return (latest - earliest) / abs(earliest)


def create_error_result(
    error_message: str,
    score: float = 0,
    max_score: float = 0,
    **kwargs
) -> dict:
    """
    Create a standardized error result dictionary.

    Args:
        error_message: Description of the error
        score: Score to return (default 0)
        max_score: Maximum possible score (default 0)
        **kwargs: Additional fields to include in result

    Returns:
        Standardized error dictionary
    """
    result = {
        "score": score,
        "max_score": max_score,
        "details": error_message,
    }
    result.update(kwargs)
    return result
