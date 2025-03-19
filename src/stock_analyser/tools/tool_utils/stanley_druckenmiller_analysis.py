import statistics
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
from stock_analyser.tools.tool_utils.news_sentiment_util import get_news_sentiment_scores, get_news

today = datetime.now().strftime("%Y-%m-%d")

def get_ticker(ticker: str):
    """
    Get a yfinance Ticker object for the given ticker symbol.
    
    Args:
        ticker (str): The stock ticker symbol
        
    Returns:
        yf.Ticker: A yfinance Ticker object
        
    Raises:
        Exception: If ticker data cannot be retrieved
    """
    try:
        return yf.Ticker(ticker.upper())
    except Exception as e:
        raise Exception(f"Failed to get ticker data for {ticker}: {str(e)}")
    

def safe_get_row(df: pd.DataFrame, row_name: str, alternative_names=None):
    """
    Safely get a row from a DataFrame, handling KeyError and empty DataFrames.
    
    Args:
        df (pd.DataFrame): The DataFrame to get the row from
        row_name (str): The name of the row to get
        alternative_names (list, optional): Alternative names to try if row_name is not found
        
    Returns:
        pd.Series or None: The row data or None if not found
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


def filter_valid_values(series):
    """
    Filter a series to only include valid numeric values (not None or NaN).
    
    Args:
        series: A pandas Series or list-like object
        
    Returns:
        list: A list of valid numeric values
    """
    if series is None:
        return []
    
    if isinstance(series, pd.Series):
        return [val for val in series if val is not None and not np.isnan(val)]
    else:
        return [val for val in series if val is not None and not np.isnan(val)]
    


def analyze_growth_and_momentum(ticker: str, prices: pd.DataFrame):
    """
    Evaluate:
      - Revenue Growth (YoY)
      - EPS Growth (YoY)
      - Price Momentum
    """

    company_ticker = get_ticker(ticker)

    details = []
    raw_score = 0

    #
    # 1. Revenue Growth
    #

    financials = company_ticker.financials
    if financials.empty:
        return {"score": 0, "max_score": 0, "details": "No financial data available"}
        
    revenue_df = safe_get_row(financials, "Total Revenue", ["Revenue", "Revenues"])
    
    if revenue_df is None:
        details.append("Revenue data not available in financial statements.")
    else:
        revenues = filter_valid_values(revenue_df)
        
    if len(revenues) >= 2:
        try:
            # Check if overall revenue grew from first to last
            # Note: yfinance data is typically in reverse chronological order (newest first)
            latest = revenues[0]
            earliest = None  # Initialize to None

            # Loop backwards to find a valid (non-zero) earliest revenue
            for i in range(len(revenues) - 1, -1, -1):
                if revenues[i] != 0:
                    earliest = revenues[i]
                    break

            if earliest > 0:
                rev_growth = (latest - earliest) / abs(earliest)
                if rev_growth > 0.3:
                    raw_score += 3
                    details.append(f"Strong revenue growth: {(rev_growth*100):.1f}% over the full period.")
                elif rev_growth > 0.15:
                    raw_score += 2
                    details.append(f"Moderate revenue growth: {(rev_growth*100):.1f}% over the full period.")
                elif rev_growth > 0.05:
                    raw_score += 1
                    details.append(f"Slight growth: ({(rev_growth*100):.1f}%).")
                else:
                    details.append(f"No growth or decline in revenue: {(rev_growth*100):.1f}%.")

            else:
                details.append("Older revenue is zero/negative; can't compute revenue growth.")
        except Exception as e:
            details.append(f"Error calculating revenue growth: {str(e)}")
    else:
        details.append("Not enough revenue data to calculate growth.")


    #
    # 2. EPS Growth
    #

    eps_df = safe_get_row(financials, "Diluted EPS", ["EPS", "Earnings Per Share"])

    if eps_df is None:
        details.append("EPS data not available in financial statements.")
    else:
        eps_values = filter_valid_values(eps_df)

    if len(eps_values) >= 2:
        try:
            latest = eps_values[0]
            earliest = None

            for i in range(len(eps_values) - 1, -1, -1):
                if eps_values[i] != 0:
                    earliest = eps_values[i]
                    break

            if abs(earliest) > 0:
                eps_growth = (latest - earliest) / abs(earliest)
                if eps_growth > 0.3:
                    raw_score += 3
                    details.append(f"Strong EPS growth: {(eps_growth*100):.1f}% over the full period.")
                elif eps_growth > 0.15:
                    raw_score += 2
                    details.append(f"Moderate EPS growth: {(eps_growth*100):.1f}% over the full period.")
                elif eps_growth > 0.05:
                    raw_score += 1
                    details.append(f"Slight growth: ({(eps_growth*100):.1f}%).")
                else:
                    details.append(f"Minimal/negative EPS growth: {(eps_growth*100):.1f}%.")

            else:
                details.append("Older EPS is zero/negative; can't compute EPS growth.")
        except Exception as e:
            details.append(f"Error calculating EPS growth: {str(e)}")
    else:
        details.append("Not enough EPS data to calculate growth.")
        
    
    #
    # 3. Price Momentum
    #
    # Up to 3 points for strong momentum

    if not prices.empty and len(prices) >= 30:
        sorted_prices = prices.sort_index()
        close_prices = sorted_prices["Close"]

        if len(close_prices) >= 2:
            start_price = close_prices.iloc[0]
            end_price = close_prices.iloc[-1]

            if start_price > 0:
                pct_change = (end_price - start_price) / abs(start_price)

                if pct_change > 0.5:
                    raw_score += 3
                    details.append(f"Very strong price momentum: {(pct_change*100):.1f}% over the full period.")
                elif pct_change > 0.2:
                    raw_score += 2
                    details.append(f"Moderate price momentum: {(pct_change*100):.1f}% over the full period.")
                elif pct_change > 0.0:
                    raw_score += 1
                    details.append(f"Slight positive momentum: {(pct_change*100):.1f}% over the full period.")
                else:
                    details.append(f"Negative price momentum: {(pct_change*100):.1f}%.")

            else:
                details.append("Start price is zero/negative; can't compute price momentum.")

        else:
            details.append("Insufficient price data to calculate momentum.")

    else:
        details.append("Not enough price data to calculate momentum.")
                           
    # We assigned up to 3 points each for:
    #   revenue growth, eps growth, momentum
    # => max raw_score = 9
    # Scale to 0–10
    final_score = min(10, (raw_score / 9) * 10)

    return {"score": final_score, "details": "; ".join(details)}


def analyze_insider_activity(insider_trades: pd.DataFrame) -> dict:
    """
    Simple insider-trade analysis:
      - If there's heavy insider buying, we nudge the score up.
      - If there's mostly selling, we reduce it.
      - Otherwise, neutral.
    """

    # Default is neutral (5/10).
    score = 5
    details = []

    if insider_trades.empty:
        details.append("No insider trades data available; defaulting to neutral.")
        return {"score": score, "details": "; ".join(details)}
    
    buys, sells = insider_trades["Trans"].iloc[0], insider_trades["Trans"].iloc[1]
    
    total = buys + sells

    if total == 0:
        details.append("No insider trades data available; defaulting to neutral.")
        return {"score": score, "details": "; ".join(details)}
    
    buy_ratio = buys / total
    
    if buy_ratio > 0.7:
        # Heavy buying => +3 points from the neutral 5 => 8
        score = 8
        details.append(f"Heavy insider buying: {buys} buys vs. {sells} sells")
    elif buy_ratio > 0.4:
        # Moderate buying => +1 => 6
        score = 6
        details.append(f"Moderate insider buying: {buys} buys vs. {sells} sells")
    else:
        # Low insider buying => -1 => 4
        score = 4
        details.append(f"Mostly insider selling: {buys} buys vs. {sells} sells")

    return {"score": score, "details": "; ".join(details)}
    


def analyze_sentiment(ticker: str) -> dict:
    """
    Analyze the sentiment of a stock.
    """
        
    sentiment_score = get_news_sentiment_scores(ticker, 7)

    if not sentiment_score:
        return {"score": 5, "details": "No sentiment data; defaulting to neutral sentiment"}

    details = []
    if sentiment_score < -30:
        # More than 30% negative => somewhat bearish => 3/10
        score = 2
        details.append(f"High proportion of negative headlines: {sentiment_score}")
    elif sentiment_score < 0:
        # Some negativity => 6/10
        score = 4
        details.append(f"Some negative headlines: {sentiment_score}")
    elif sentiment_score > 30:
        # More than 30% positive => somewhat bullish => 7/10
        score = 6
        details.append(f"High proportion of positive headlines: {sentiment_score}")
    else:
        # Mostly positive => 8/10
        score = 8
        details.append("Mostly positive/neutral headlines")

    return {"score": score, "details": "; ".join(details)}


def analyze_risk_reward(ticker: str, prices: pd.DataFrame):
    """
    Assesses risk via:
      - Debt-to-Equity
      - Price Volatility
    Aims for strong upside with contained downside.
    """

    company_ticker = get_ticker(ticker)
    if prices.empty:
        return {"score": 0, "details": "Insufficient data for risk-reward analysis"}
    
    details = []

    raw_score = 0  # We'll accumulate up to 6 raw points, then scale to 0-10

    #
    # 1. Debt-to-Equity
    #

    balance_sheet = company_ticker.balance_sheet
    if balance_sheet.empty:
        return {"score": 0, "max_score": 0, "details": "No financial data available"}
    
    debt_values = safe_get_row(balance_sheet, "Total Debt", ["Current Debt"])
    equity_values = safe_get_row(balance_sheet, "Stockholders Equity")

    if not debt_values.empty and not equity_values.empty and len(debt_values) == len(equity_values) and len(debt_values) > 0:
        recent_debt = debt_values.iloc[0]
        recent_equity = equity_values.iloc[0] if equity_values.iloc[0] else 1e-9
        de_ratio = recent_debt / recent_equity

        if de_ratio < 0.3:
            raw_score += 3
            details.append(f"Low debt-to-equity ratio: {de_ratio:.2f}")
        elif de_ratio < 0.7:
            raw_score += 2
            details.append(f"Moderate debt-to-equity ratio: {de_ratio:.2f}")
        elif de_ratio < 1.5:
            raw_score += 1
            details.append(f"Somewhat high debt-to-equity ratio: {de_ratio:.2f}")
        else:
            details.append(f"High debt-to-equity ratio: {de_ratio:.2f}")

    else:
        details.append("No consistent debt/equity data available.")

    
    #
    # 2. Price Volatility
    #
    if not prices.empty and len(prices) >= 10:
        sorted_prices = prices.sort_index()
        close_prices = sorted_prices["Close"]

        if len(close_prices) > 10:
            daily_returns = []
            for i in range(1, len(close_prices)):
                prev_close = close_prices.iloc[i - 1]
                if prev_close > 0:
                    daily_returns.append((close_prices.iloc[i] - prev_close) / prev_close)
            if daily_returns:
                stdev = statistics.pstdev(daily_returns)  # population stdev
                if stdev < 0.01:
                    raw_score += 3
                    details.append(f"Low volatility: daily returns stdev {stdev:.2%}")
                elif stdev < 0.02:
                    raw_score += 2
                    details.append(f"Moderate volatility: daily returns stdev {stdev:.2%}")
                elif stdev < 0.04:
                    raw_score += 1
                    details.append(f"High volatility: daily returns stdev {stdev:.2%}")
                else:
                    details.append(f"Very high volatility: daily returns stdev {stdev:.2%}")
            else:
                details.append("Insufficient daily returns data for volatility calc.")
        else:
            details.append("Not enough close-price data points for volatility analysis.")
    else:
        details.append("Not enough price data for volatility analysis.")

    # raw_score out of 6 => scale to 0–10
    final_score = min(10, (raw_score / 6) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_druckenmiller_valuation(ticker: str, market_cap: float):
    """
    Druckenmiller is willing to pay up for growth, but still checks:
      - P/E
      - P/FCF
      - EV/EBIT
      - EV/EBITDA
    Each can yield up to 2 points => max 8 raw points => scale to 0–10.
    """

    company_ticker = get_ticker(ticker)

    financials = company_ticker.financials
    cash_flow = company_ticker.cashflow
    balance_sheet = company_ticker.balance_sheet
    if financials.empty or cash_flow.empty or balance_sheet.empty:
        return {"score": 0, "max_score": 0, "details": "No financial data available"}

    if not market_cap:
        return {"score": 0, "details": "Insufficient data to perform valuation"}

    details = []
    raw_score = 0

    # Gather needed data
    net_incomes = filter_valid_values(safe_get_row(financials, "Net Income"))
    fcf_values = filter_valid_values(safe_get_row(cash_flow, "Free Cash Flow"))
    ebit_values = filter_valid_values(safe_get_row(financials, "EBIT"))
    ebitda_values = filter_valid_values(safe_get_row(financials, "EBITDA"))

    # For EV calculation, let's get the most recent total_debt & cash
    debt_values = filter_valid_values(safe_get_row(balance_sheet, "Total Debt", ["Current Debt"]))
    cash_values = filter_valid_values(safe_get_row(financials, "Cash and Cash Equivalents")) # Consider adding alternative names if necessary.
    recent_debt = debt_values[0] if debt_values else 0
    recent_cash = cash_values[0] if cash_values else 0

    enterprise_value = market_cap + recent_debt - recent_cash

    # 1) P/E
    recent_net_income = net_incomes[0] if net_incomes else None
    if recent_net_income and recent_net_income > 0:
        pe = market_cap / recent_net_income
        pe_points = 0
        if pe < 15:
            pe_points = 2
            details.append(f"Attractive P/E: {pe:.2f}")
        elif pe < 25:
            pe_points = 1
            details.append(f"Fair P/E: {pe:.2f}")
        else:
            details.append(f"High or Very high P/E: {pe:.2f}")
        raw_score += pe_points
    else:
        details.append("No positive net income for P/E calculation")

    # 2) P/FCF
    recent_fcf = fcf_values[0] if fcf_values else None
    if recent_fcf and recent_fcf > 0:
        pfcf = market_cap / recent_fcf
        pfcf_points = 0
        if pfcf < 15:
            pfcf_points = 2
            details.append(f"Attractive P/FCF: {pfcf:.2f}")
        elif pfcf < 25:
            pfcf_points = 1
            details.append(f"Fair P/FCF: {pfcf:.2f}")
        else:
            details.append(f"High/Very high P/FCF: {pfcf:.2f}")
        raw_score += pfcf_points
    else:
        details.append("No positive free cash flow for P/FCF calculation")

    # 3) EV/EBIT
    recent_ebit = ebit_values[0] if ebit_values else None
    if enterprise_value > 0 and recent_ebit and recent_ebit > 0:
        ev_ebit = enterprise_value / recent_ebit
        ev_ebit_points = 0
        if ev_ebit < 15:
            ev_ebit_points = 2
            details.append(f"Attractive EV/EBIT: {ev_ebit:.2f}")
        elif ev_ebit < 25:
            ev_ebit_points = 1
            details.append(f"Fair EV/EBIT: {ev_ebit:.2f}")
        else:
            details.append(f"High EV/EBIT: {ev_ebit:.2f}")
        raw_score += ev_ebit_points
    else:
        details.append("No valid EV/EBIT because EV <= 0 or EBIT <= 0")

    # 4) EV/EBITDA
    recent_ebitda = ebitda_values[0] if ebitda_values else None
    if enterprise_value > 0 and recent_ebitda and recent_ebitda > 0:
        ev_ebitda = enterprise_value / recent_ebitda
        ev_ebitda_points = 0
        if ev_ebitda < 10:
            ev_ebitda_points = 2
            details.append(f"Attractive EV/EBITDA: {ev_ebitda:.2f}")
        elif ev_ebitda < 18:
            ev_ebitda_points = 1
            details.append(f"Fair EV/EBITDA: {ev_ebitda:.2f}")
        else:
            details.append(f"High EV/EBITDA: {ev_ebitda:.2f}")
        raw_score += ev_ebitda_points
    else:
        details.append("No valid EV/EBITDA because EV <= 0 or EBITDA <= 0")

    # We have up to 2 points for each of the 4 metrics => 8 raw points max
    # Scale raw_score to 0–10
    final_score = min(10, (raw_score / 8) * 10)

    return {"score": final_score, "details": "; ".join(details)}
        



def calculate_druckenmiller_data(ticker: str):
    """
    Calculate the data for the Druckenmiller analysis.
    """

    company_ticker = get_ticker(ticker)
    market_cap = company_ticker.info['marketCap']

    insider_trades = company_ticker.insider_purchases

    prices = company_ticker.history(start="2022-01-01", end=today)

    growth_momentum_analysis = analyze_growth_and_momentum(ticker, prices)

    sentiment_analysis = analyze_sentiment(ticker)

    insider_activity = analyze_insider_activity(insider_trades)

    risk_reward_analysis = analyze_risk_reward(ticker, prices)

    valuation_analysis = analyze_druckenmiller_valuation(ticker, market_cap)

    # Combine partial scores with weights typical for Druckenmiller:
    #   35% Growth/Momentum, 20% Risk/Reward, 20% Valuation,
    #   15% Sentiment, 10% Insider Activity = 100%
    total_score = (
        growth_momentum_analysis["score"] * 0.35
        + risk_reward_analysis["score"] * 0.20
        + valuation_analysis["score"] * 0.20
        + sentiment_analysis["score"] * 0.15
        + insider_activity["score"] * 0.10
    )

    max_possible_score = 10

    # Simple bullish/neutral/bearish signal
    if total_score >= 7.5:
        signal = "bullish"
    elif total_score <= 4.5:
        signal = "bearish"
    else:
        signal = "neutral"

    analysis_data = {
        "signal": signal,
        "score": total_score,
        "max_score": max_possible_score,
        "growth_momentum_analysis": growth_momentum_analysis,
        "sentiment_analysis": sentiment_analysis,
        "insider_activity": insider_activity,
        "risk_reward_analysis": risk_reward_analysis,
        "valuation_analysis": valuation_analysis,
    }

    return analysis_data


if __name__ == "__main__":
    ticker = "AAPL"
    print(calculate_druckenmiller_data(ticker))
