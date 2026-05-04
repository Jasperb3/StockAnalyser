"""
Stanley Druckenmiller investment analysis module.

Analyzes stocks using Druckenmiller's momentum and growth-focused approach:
- Revenue and EPS growth momentum
- Price momentum trends
- Insider activity signals
- Risk/reward assessment (debt levels, volatility)
- Growth-adjusted valuation multiples
"""

from typing import Dict, Any
import statistics
import pandas as pd
from datetime import datetime
from stock_analyser.tools.tool_utils.news_sentiment_util import (
    get_news_sentiment_scores,
)
from stock_analyser.tools.tool_utils.analysis_helpers import (
    get_ticker,
    safe_get_row,
    filter_valid_values,
    safe_divide,
    create_error_result,
)
from stock_analyser.tools.tool_utils.analysis_constants import (
    REVENUE_GROWTH_EXCEPTIONAL,
    REVENUE_GROWTH_GOOD,
    REVENUE_GROWTH_MODERATE,
    EPS_GROWTH_EXCEPTIONAL,
    EPS_GROWTH_GOOD,
    EPS_GROWTH_MODERATE,
    DEBT_TO_EQUITY_CONSERVATIVE,
    DEBT_TO_EQUITY_MODERATE,
    DEBT_TO_EQUITY_HIGH,
    PE_ATTRACTIVE,
    PE_FAIR,
    PFCF_ATTRACTIVE,
    PFCF_FAIR,
    EV_EBIT_ATTRACTIVE,
    EV_EBIT_FAIR,
    EV_EBITDA_ATTRACTIVE,
    EV_EBITDA_FAIR,
    PRICE_HISTORY_START_DATE,
)
from stock_analyser.utils.convert_currency import convert_currency

# Druckenmiller-specific constants
DRUCKENMILLER_NEWS_ARTICLES = 7

# Druckenmiller weighting for final score
DRUCKENMILLER_WEIGHTS = {
    "growth_momentum": 0.35,
    "risk_reward": 0.20,
    "valuation": 0.20,
    "sentiment": 0.15,
    "insider_activity": 0.10,
}

today = datetime.now().strftime("%Y-%m-%d")


def analyze_growth_and_momentum(ticker: str, prices: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate revenue growth, EPS growth, and price momentum.

    Args:
        ticker: Stock ticker symbol
        prices: Historical price DataFrame

    Returns:
        Analysis results with score and details
    """
    try:
        company_ticker = get_ticker(ticker)
    except Exception as e:
        return create_error_result(f"Failed to get ticker data: {str(e)}")

    details = []
    raw_score = 0

    # 1. Revenue Growth
    try:
        financials = company_ticker.financials
        if financials is None or financials.empty:
            return create_error_result("No financial data available")
    except Exception as e:
        return create_error_result(f"Error retrieving financial data: {str(e)}")

    revenue_df = safe_get_row(financials, "Total Revenue", ["Revenue", "Revenues"])

    if revenue_df is not None and not revenue_df.empty:
        revenues = filter_valid_values(revenue_df)

        if len(revenues) >= 2:
            latest = revenues[0]
            earliest = None

            # Find valid earliest revenue (non-zero)
            for i in range(len(revenues) - 1, -1, -1):
                if revenues[i] != 0:
                    earliest = revenues[i]
                    break

            if earliest and earliest > 0:
                rev_growth = safe_divide(latest - earliest, abs(earliest))

                if rev_growth > REVENUE_GROWTH_EXCEPTIONAL:
                    raw_score += 3
                    details.append(
                        f"Strong revenue growth: {(rev_growth * 100):.1f}% over the full period."
                    )
                elif rev_growth > REVENUE_GROWTH_GOOD:
                    raw_score += 2
                    details.append(
                        f"Moderate revenue growth: {(rev_growth * 100):.1f}% over the full period."
                    )
                elif rev_growth > REVENUE_GROWTH_MODERATE:
                    raw_score += 1
                    details.append(f"Slight growth: ({(rev_growth * 100):.1f}%).")
                else:
                    details.append(
                        f"No growth or decline in revenue: {(rev_growth * 100):.1f}%."
                    )
            else:
                details.append(
                    "Older revenue is zero/negative; can't compute revenue growth."
                )
        else:
            details.append("Not enough revenue data to calculate growth.")
    else:
        details.append("Revenue data not available in financial statements.")

    # 2. EPS Growth
    eps_df = safe_get_row(financials, "Diluted EPS", ["EPS", "Earnings Per Share"])

    if eps_df is not None and not eps_df.empty:
        eps_values = filter_valid_values(eps_df)

        if len(eps_values) >= 2:
            latest = eps_values[0]
            earliest = None

            for i in range(len(eps_values) - 1, -1, -1):
                if eps_values[i] != 0:
                    earliest = eps_values[i]
                    break

            if earliest and abs(earliest) > 0:
                eps_growth = safe_divide(latest - earliest, abs(earliest))

                if eps_growth > EPS_GROWTH_EXCEPTIONAL:
                    raw_score += 3
                    details.append(
                        f"Strong EPS growth: {(eps_growth * 100):.1f}% over the full period."
                    )
                elif eps_growth > EPS_GROWTH_GOOD:
                    raw_score += 2
                    details.append(
                        f"Moderate EPS growth: {(eps_growth * 100):.1f}% over the full period."
                    )
                elif eps_growth > EPS_GROWTH_MODERATE:
                    raw_score += 1
                    details.append(f"Slight growth: ({(eps_growth * 100):.1f}%).")
                else:
                    details.append(
                        f"Minimal/negative EPS growth: {(eps_growth * 100):.1f}%."
                    )
            else:
                details.append("Older EPS is zero/negative; can't compute EPS growth.")
        else:
            details.append("Not enough EPS data to calculate growth.")
    else:
        details.append("EPS data not available in financial statements.")

    # 3. Price Momentum
    if prices is not None and not prices.empty and len(prices) >= 30:
        sorted_prices = prices.sort_index()
        close_prices = sorted_prices["Close"]

        if len(close_prices) >= 2:
            start_price = close_prices.iloc[0]
            end_price = close_prices.iloc[-1]

            if start_price > 0:
                pct_change = safe_divide(end_price - start_price, abs(start_price))

                if pct_change > 0.5:
                    raw_score += 3
                    details.append(
                        f"Very strong price momentum: {(pct_change * 100):.1f}% over the full period."
                    )
                elif pct_change > 0.2:
                    raw_score += 2
                    details.append(
                        f"Moderate price momentum: {(pct_change * 100):.1f}% over the full period."
                    )
                elif pct_change > 0.0:
                    raw_score += 1
                    details.append(
                        f"Slight positive momentum: {(pct_change * 100):.1f}% over the full period."
                    )
                else:
                    details.append(
                        f"Negative price momentum: {(pct_change * 100):.1f}%."
                    )
            else:
                details.append(
                    "Start price is zero/negative; can't compute price momentum."
                )
        else:
            details.append("Insufficient price data to calculate momentum.")
    else:
        details.append("Not enough price data to calculate momentum.")

    # Max raw_score = 9 (3 + 3 + 3), scale to 0-10
    final_score = min(10, safe_divide(raw_score * 10, 9, default=0))

    return {"score": final_score, "details": "; ".join(details)}


def analyze_insider_activity(insider_trades: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze insider trading activity.

    Args:
        insider_trades: DataFrame with insider trading data

    Returns:
        Analysis results with score and details
    """
    score = 5  # Default neutral
    details = []

    if insider_trades is None or insider_trades.empty:
        details.append("No insider trades data available; defaulting to neutral.")
        return {"score": score, "details": "; ".join(details)}

    if len(insider_trades) < 2:
        details.append("Insufficient insider trades data; defaulting to neutral.")
        return {"score": score, "details": "; ".join(details)}

    buys = insider_trades["Trans"].iloc[0]
    sells = insider_trades["Trans"].iloc[1]
    total = buys + sells

    if total == 0:
        details.append("No insider trades data available; defaulting to neutral.")
        return {"score": score, "details": "; ".join(details)}

    buy_ratio = safe_divide(buys, total)

    if buy_ratio > 0.7:
        score = 8
        details.append(f"Heavy insider buying: {buys} buys vs. {sells} sells")
    elif buy_ratio > 0.4:
        score = 6
        details.append(f"Moderate insider buying: {buys} buys vs. {sells} sells")
    else:
        score = 4
        details.append(f"Mostly insider selling: {buys} buys vs. {sells} sells")

    return {"score": score, "details": "; ".join(details)}


def analyze_sentiment(ticker: str) -> Dict[str, Any]:
    """
    Analyze news sentiment for the stock.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Sentiment analysis results
    """
    try:
        sentiment_score = get_news_sentiment_scores(ticker, DRUCKENMILLER_NEWS_ARTICLES)

        if not sentiment_score:
            return {
                "score": 5,
                "details": "No sentiment data; defaulting to neutral sentiment",
            }

        details = []
        if sentiment_score < -30:
            score = 2
            details.append(f"High proportion of negative headlines: {sentiment_score}")
        elif sentiment_score < 0:
            score = 4
            details.append(f"Some negative headlines: {sentiment_score}")
        elif sentiment_score > 30:
            score = 6
            details.append(f"High proportion of positive headlines: {sentiment_score}")
        else:
            score = 8
            details.append("Mostly positive/neutral headlines")

        return {"score": score, "details": "; ".join(details)}
    except Exception as e:
        return {"score": 5, "details": f"Error analyzing sentiment: {str(e)}"}


def analyze_risk_reward(ticker: str, prices: pd.DataFrame) -> Dict[str, Any]:
    """
    Assess risk via debt-to-equity and price volatility.

    Args:
        ticker: Stock ticker symbol
        prices: Historical price DataFrame

    Returns:
        Risk/reward analysis results
    """
    try:
        company_ticker = get_ticker(ticker)
    except Exception as e:
        return create_error_result(f"Failed to get ticker data: {str(e)}")

    if prices is None or prices.empty:
        return create_error_result("Insufficient price data for risk-reward analysis")

    details = []
    raw_score = 0

    # 1. Debt-to-Equity
    try:
        balance_sheet = company_ticker.balance_sheet
        if balance_sheet is None or balance_sheet.empty:
            return create_error_result("No balance sheet data available")
    except Exception as e:
        return create_error_result(f"Error retrieving balance sheet data: {str(e)}")

    debt_values = safe_get_row(balance_sheet, "Total Debt", ["Current Debt"])
    equity_values = safe_get_row(balance_sheet, "Stockholders Equity")

    if (
        debt_values is not None
        and not debt_values.empty
        and equity_values is not None
        and not equity_values.empty
        and len(debt_values) > 0
        and len(equity_values) > 0
    ):
        recent_debt = debt_values.iloc[0]
        recent_equity = equity_values.iloc[0] if equity_values.iloc[0] else 1e-9
        de_ratio = safe_divide(recent_debt, recent_equity)

        if de_ratio < DEBT_TO_EQUITY_CONSERVATIVE:
            raw_score += 3
            details.append(f"Low debt-to-equity ratio: {de_ratio:.2f}")
        elif de_ratio < DEBT_TO_EQUITY_MODERATE:
            raw_score += 2
            details.append(f"Moderate debt-to-equity ratio: {de_ratio:.2f}")
        elif de_ratio < DEBT_TO_EQUITY_HIGH:
            raw_score += 1
            details.append(f"Somewhat high debt-to-equity ratio: {de_ratio:.2f}")
        else:
            details.append(f"High debt-to-equity ratio: {de_ratio:.2f}")
    else:
        details.append("No consistent debt/equity data available.")

    # 2. Price Volatility
    if not prices.empty and len(prices) >= 10:
        sorted_prices = prices.sort_index()
        close_prices = sorted_prices["Close"]

        if len(close_prices) > 10:
            daily_returns = []
            for i in range(1, len(close_prices)):
                prev_close = close_prices.iloc[i - 1]
                if prev_close > 0:
                    daily_returns.append(
                        safe_divide(close_prices.iloc[i] - prev_close, prev_close)
                    )

            if daily_returns:
                stdev = statistics.pstdev(daily_returns)
                if stdev < 0.01:
                    raw_score += 3
                    details.append(f"Low volatility: daily returns stdev {stdev:.2%}")
                elif stdev < 0.02:
                    raw_score += 2
                    details.append(
                        f"Moderate volatility: daily returns stdev {stdev:.2%}"
                    )
                elif stdev < 0.04:
                    raw_score += 1
                    details.append(f"High volatility: daily returns stdev {stdev:.2%}")
                else:
                    details.append(
                        f"Very high volatility: daily returns stdev {stdev:.2%}"
                    )
            else:
                details.append("Insufficient daily returns data for volatility calc.")
        else:
            details.append(
                "Not enough close-price data points for volatility analysis."
            )
    else:
        details.append("Not enough price data for volatility analysis.")

    # raw_score out of 6, scale to 0-10
    final_score = min(10, safe_divide(raw_score * 10, 6, default=0))
    return {"score": final_score, "details": "; ".join(details)}


def analyze_druckenmiller_valuation(ticker: str) -> Dict[str, Any]:
    """
    Druckenmiller valuation using P/E, P/FCF, EV/EBIT, EV/EBITDA.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Valuation analysis results
    """
    try:
        company_ticker = get_ticker(ticker)
        exchange_rate = convert_currency(ticker)
    except Exception as e:
        return create_error_result(f"Failed to get ticker data: {str(e)}")

    try:
        # Apply exchange rate only to numeric columns
        financials_raw = company_ticker.financials
        cash_flow_raw = company_ticker.cashflow
        balance_sheet_raw = company_ticker.balance_sheet

        if financials_raw is None or financials_raw.empty:
            return create_error_result("No financial data available")
        if cash_flow_raw is None or cash_flow_raw.empty:
            return create_error_result("No cash flow data available")
        if balance_sheet_raw is None or balance_sheet_raw.empty:
            return create_error_result("No balance sheet data available")

        # Only apply exchange rate to numeric data
        financials = financials_raw.apply(
            lambda x: x * exchange_rate if pd.api.types.is_numeric_dtype(x) else x
        )
        cash_flow = cash_flow_raw.apply(
            lambda x: x * exchange_rate if pd.api.types.is_numeric_dtype(x) else x
        )
        balance_sheet = balance_sheet_raw.apply(
            lambda x: x * exchange_rate if pd.api.types.is_numeric_dtype(x) else x
        )

        market_cap = company_ticker.info.get("marketCap", None)
        if not market_cap:
            return create_error_result("Market cap data not available")

    except Exception as e:
        return create_error_result(f"Error retrieving financial data: {str(e)}")

    details = []
    raw_score = 0

    # Gather needed data
    net_incomes = filter_valid_values(safe_get_row(financials, "Net Income"))
    fcf_values = filter_valid_values(safe_get_row(cash_flow, "Free Cash Flow"))
    ebit_values = filter_valid_values(safe_get_row(financials, "EBIT"))
    ebitda_values = filter_valid_values(safe_get_row(financials, "EBITDA"))

    # For EV calculation
    debt_values = filter_valid_values(
        safe_get_row(balance_sheet, "Total Debt", ["Current Debt"])
    )
    cash_values = filter_valid_values(
        safe_get_row(balance_sheet, "Cash And Cash Equivalents")
    )

    recent_debt = debt_values[0] if debt_values else 0
    recent_cash = cash_values[0] if cash_values else 0

    enterprise_value = market_cap + recent_debt - recent_cash

    # 1. P/E
    recent_net_income = net_incomes[0] if net_incomes else None
    if recent_net_income and recent_net_income > 0:
        pe = safe_divide(market_cap, recent_net_income)

        if pe < PE_ATTRACTIVE:
            raw_score += 2
            details.append(f"Attractive P/E: {pe:.2f}")
        elif pe < PE_FAIR:
            raw_score += 1
            details.append(f"Fair P/E: {pe:.2f}")
        else:
            details.append(f"High or Very high P/E: {pe:.2f}")
    else:
        details.append("No positive net income for P/E calculation")

    # 2. P/FCF
    recent_fcf = fcf_values[0] if fcf_values else None
    if recent_fcf and recent_fcf > 0:
        pfcf = safe_divide(market_cap, recent_fcf)

        if pfcf < PFCF_ATTRACTIVE:
            raw_score += 2
            details.append(f"Attractive P/FCF: {pfcf:.2f}")
        elif pfcf < PFCF_FAIR:
            raw_score += 1
            details.append(f"Fair P/FCF: {pfcf:.2f}")
        else:
            details.append(f"High/Very high P/FCF: {pfcf:.2f}")
    else:
        details.append("No positive free cash flow for P/FCF calculation")

    # 3. EV/EBIT
    recent_ebit = ebit_values[0] if ebit_values else None
    if enterprise_value > 0 and recent_ebit and recent_ebit > 0:
        ev_ebit = safe_divide(enterprise_value, recent_ebit)

        if ev_ebit < EV_EBIT_ATTRACTIVE:
            raw_score += 2
            details.append(f"Attractive EV/EBIT: {ev_ebit:.2f}")
        elif ev_ebit < EV_EBIT_FAIR:
            raw_score += 1
            details.append(f"Fair EV/EBIT: {ev_ebit:.2f}")
        else:
            details.append(f"High EV/EBIT: {ev_ebit:.2f}")
    else:
        details.append("No valid EV/EBIT because EV <= 0 or EBIT <= 0")

    # 4. EV/EBITDA
    recent_ebitda = ebitda_values[0] if ebitda_values else None
    if enterprise_value > 0 and recent_ebitda and recent_ebitda > 0:
        ev_ebitda = safe_divide(enterprise_value, recent_ebitda)

        if ev_ebitda < EV_EBITDA_ATTRACTIVE:
            raw_score += 2
            details.append(f"Attractive EV/EBITDA: {ev_ebitda:.2f}")
        elif ev_ebitda < EV_EBITDA_FAIR:
            raw_score += 1
            details.append(f"Fair EV/EBITDA: {ev_ebitda:.2f}")
        else:
            details.append(f"High EV/EBITDA: {ev_ebitda:.2f}")
    else:
        details.append("No valid EV/EBITDA because EV <= 0 or EBITDA <= 0")

    # Max raw_score = 8 (2 per metric), scale to 0-10
    final_score = min(10, safe_divide(raw_score * 10, 8, default=0))

    return {"score": final_score, "details": "; ".join(details)}


def calculate_druckenmiller_data(ticker: str) -> Dict[str, Any]:
    """
    Calculate comprehensive Druckenmiller-style analysis.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Complete analysis with signal, score, and component analyses
    """
    try:
        company_ticker = get_ticker(ticker)
    except Exception as e:
        return create_error_result(f"Failed to get ticker data: {str(e)}")

    try:
        insider_trades = company_ticker.insider_purchases
    except Exception:
        insider_trades = pd.DataFrame()

    try:
        prices = company_ticker.history(start=PRICE_HISTORY_START_DATE, end=today)
    except Exception:
        prices = pd.DataFrame()

    growth_momentum_analysis = analyze_growth_and_momentum(ticker, prices)
    sentiment_analysis = analyze_sentiment(ticker)
    insider_activity = analyze_insider_activity(insider_trades)
    risk_reward_analysis = analyze_risk_reward(ticker, prices)
    valuation_analysis = analyze_druckenmiller_valuation(ticker)

    # Combine scores with Druckenmiller weights
    total_score = (
        growth_momentum_analysis["score"] * DRUCKENMILLER_WEIGHTS["growth_momentum"]
        + risk_reward_analysis["score"] * DRUCKENMILLER_WEIGHTS["risk_reward"]
        + valuation_analysis["score"] * DRUCKENMILLER_WEIGHTS["valuation"]
        + sentiment_analysis["score"] * DRUCKENMILLER_WEIGHTS["sentiment"]
        + insider_activity["score"] * DRUCKENMILLER_WEIGHTS["insider_activity"]
    )

    max_possible_score = 10

    # Generate signal
    if total_score >= 7.5:
        signal = "bullish"
    elif total_score <= 4.5:
        signal = "bearish"
    else:
        signal = "neutral"

    analysis_data = {
        "ticker": ticker,
        "signal": signal,
        "score": round(total_score, 2),
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
