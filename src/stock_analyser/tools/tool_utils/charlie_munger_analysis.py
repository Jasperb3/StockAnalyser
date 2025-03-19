from stock_analyser.tools.tool_utils.news_sentiment_util import get_news_sentiment_scores
import yfinance as yf
import numpy as np
import pandas as pd


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


def analyse_moat_strength(ticker: str):
    """
    Analyze the business's competitive advantage using Munger's approach:
    - Consistent high returns on capital (ROIC)
    - Pricing power (stable/improving gross margins)
    - Low capital requirements
    - Network effects and intangible assets (R&D investments, goodwill)
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        dict: Analysis results with score and details
    """
    score = 0
    max_score = 0 # Initialize max_score
    details = []
    
    try:
        company_ticker = get_ticker(ticker)
    except Exception as e:
        return {"score": 0, "max_score": 0, "details": f"Failed to get ticker data: {str(e)}"}
    
    try:
        financials = company_ticker.financials
        if financials is None or financials.empty:
            return {"score": 0, "max_score": 0, "details": "No financial data available"}
    except Exception as e:
        return {"score": 0, "max_score": 0, "details": f"Error retrieving financial data: {str(e)}"}
    
    try:
        balance_sheet = company_ticker.balance_sheet
        if balance_sheet.empty:
            return {"score": 0, "max_score": 0, "details": "No balance sheet data available"}
    except Exception as e:
        return {"score": 0, "max_score": 0, "details": f"Error retrieving balance sheet data: {str(e)}"}
    
    try:
        cash_flow = company_ticker.cashflow
        if cash_flow.empty:
            return {"score": 0, "max_score": 0, "details": "No cash flow data available"}
    except Exception as e:
        return {"score": 0, "max_score": 0, "details": f"Error retrieving cash flow data: {str(e)}"}
    
    try:
        income_statement = company_ticker.income_stmt
        if income_statement.empty:
            return {"score": 0, "max_score": 0, "details": "No income statement data available"}
    except Exception as e:
        return {"score": 0, "max_score": 0, "details": f"Error retrieving income statement data: {str(e)}"}
    
    # 1. Return on Invested Capital (ROIC) - Munger's favorite metric
    # ROIC = Net Operating Profit After Tax (NOPAT) / Invested Capital, where NOPAT = Operating Income * (1 – Tax Rate)
    # Written another way, ROIC = (net income – dividends) / (debt + equity)

    try:
        net_income = safe_get_row(income_statement, "Net Income")
    except Exception as e:
        net_income = None

    try:
        dividends = safe_get_row(cash_flow, "Cash Dividends Paid")
    except Exception as e:
        dividends = None

    try:
        debt = safe_get_row(balance_sheet, "Total Debt")
    except Exception as e:
        debt = None

    try:
        equity = safe_get_row(balance_sheet, "Stockholders Equity")
    except Exception as e:
        equity = None

    if not net_income.empty and not dividends.empty and not debt.empty and not equity.empty:
        return_on_invested_capital = (net_income - dividends) / (debt + equity)
        roic_values = filter_valid_values(return_on_invested_capital)
   
        n = len(roic_values)
        if roic_values and n > 0:
            # Check if ROIC consistently above 15% (Munger's threshold)
            high_roic_count = sum(1 for r in roic_values if r > 0.15)
            if high_roic_count >= len(roic_values) * 0.8:  # 80% of periods show high ROIC
                score += 3
                details.append(f"Excellent ROIC: >15% in {high_roic_count}/{len(roic_values)} periods")
            elif high_roic_count >= len(roic_values) * 0.5:  # 50% of periods
                score += 2
                details.append(f"Good ROIC: >15% in {high_roic_count}/{len(roic_values)} periods")
            elif high_roic_count > 0:
                score += 1
                details.append(f"Mixed ROIC: >15% in only {high_roic_count}/{len(roic_values)} periods")
            else:
                details.append("Poor ROIC: Never exceeds 15% threshold")
    else:
        details.append(f"No ROIC data available (net_income: {net_income}, dividends: {dividends}, debt: {debt}, equity: {equity})")

    max_score += 3 # Increment by the *highest* possible score for this section
    
    # 2. Pricing power - check gross margin stability and trends

    try:
        gross_profit = safe_get_row(financials, "Gross Profit")
    except Exception as e:
        gross_profit = None

    try:
        total_revenue = safe_get_row(financials, "Total Revenue")
    except Exception as e:
        total_revenue = None

    if not gross_profit.empty and not total_revenue.empty:
        gross_margins = gross_profit / total_revenue
        gross_margin_values = [gm for gm in gross_margins if gm is not None and not np.isnan(gm)]
        

        if gross_margin_values and n > 3:
            # Munger likes stable or improving gross margins
            margin_trend = sum(1 for i in range(1, len(gross_margins)) if gross_margins.iloc[i-1] >= gross_margins.iloc[i])
            if margin_trend >= len(gross_margins) * 0.7:  # Improving in 70% of periods
                score += 2
                details.append("Strong pricing power: Gross margins consistently improving")
            elif sum(gross_margins) / len(gross_margins) > 0.3:  # Average margin > 30%
                score += 1
                details.append(f"Good pricing power: Average gross margin {sum(gross_margins)/len(gross_margins):.1%}")
            else:
                details.append("Limited pricing power: Low or declining gross margins")
    else:
        details.append(f"Insufficient gross margin data (gross_profit: {gross_profit}, total_revenue: {total_revenue})")

    max_score += 2 # Increment by the *highest* possible score for this section
        

    # 3. Capital intensity - Munger prefers low capex businesses
    
    capex_to_revenue = []

    capex = safe_get_row(cash_flow, "Capital Expenditure")

    capex_values = filter_valid_values(capex)

    revenue_values = filter_valid_values(total_revenue)

    if len(capex_values) >= 3 and len(revenue_values) >= 3:
        for i in range(len(capex_values)):
            if revenue_values[i] > 0 and not np.isnan(revenue_values[i]) and not np.isnan(capex_values[i]):
                capex_to_revenue.append(abs(capex_values[i]) / revenue_values[i])

        if capex_to_revenue:
            avg_capex_ratio = sum(capex_to_revenue) / len(capex_to_revenue)
            if avg_capex_ratio < 0.05:  # Less than 5% of revenue
                score += 2
                details.append(f"Low capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue")
            elif avg_capex_ratio < 0.10:  # Less than 10% of revenue
                score += 1
                details.append(f"Moderate capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue")
            else:
                details.append(f"High capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue")
        else:
            details.append("No capital expenditure data available")
    else:
        details.append("Insufficient data for capital intensity analysis")

    
    # 4. Intangible assets - Munger values R&D and intellectual propert

    goodwill_and_intangible_assets = safe_get_row(balance_sheet, "Goodwill And Intangible Assets")

    research_and_development = safe_get_row(financials, "Research And Development")

    g_and_i_values = filter_valid_values(goodwill_and_intangible_assets)
    r_and_d_values = filter_valid_values(research_and_development)

    if g_and_i_values and len(g_and_i_values) >= 0:
        if sum(g_and_i_values) > 0:
            score += 1
            details.append("Significant goodwill/intangible assets, suggesting brand value or IP")
        else:
            details.append("No goodwill/intangible assets")
    
    
    if r_and_d_values and len(r_and_d_values) >= 0:
        if sum(r_and_d_values) > 0:  # If company is investing in R&D
            score += 1
            details.append("Invests in R&D, building intellectual property")
        else:
            details.append("No R&D investment")

    max_score += 2 # Increment by the *highest* possible score for this section

    final_score = round(min(10, score * 10 / max_score), 1)

    return {
        "score": final_score,
        "normalized_max_score": 10,
        "details": "; ".join(details)
    }


def analyse_management_quality(ticker: str) -> dict:
    """
    Evaluate management quality using Munger's criteria:
    - Capital allocation wisdom
    - Insider ownership and transactions
    - Cash management efficiency
    - Candor and transparency
    - Long-term focus
    
    Args:
        ticker (str): Stock ticker symbol

    Returns:
        dict: Analysis results with score and details
    """
    score = 0
    max_score = 0 # Initialize max_score
    details = []


    try:
        company_ticker = get_ticker(ticker)
    except Exception as e:
        return {"score": 0, "max_score": 0, "details": f"Failed to get ticker data: {str(e)}"}
    
    try:
        balance_sheet = company_ticker.balance_sheet
        if balance_sheet is None or balance_sheet.empty:
            return {"score": 0, "max_score": 0, "details": "No balance sheet data available"}
    except Exception as e:
        return {"score": 0, "max_score": 0, "details": f"Error retrieving balance sheet data: {str(e)}"}
    
    try:
        financials = company_ticker.financials
        if financials is None or financials.empty:
            return {"score": 0, "max_score": 0, "details": "No financial data available"}
    except Exception as e:
        return {"score": 0, "max_score": 0, "details": f"Error retrieving financial data: {str(e)}"}
            
    try:
        cash_flow = company_ticker.cashflow
        if cash_flow is None or cash_flow.empty:
            return {"score": 0, "max_score": 0, "details": "No cash flow data available"}
    except Exception as e:
        return {"score": 0, "max_score": 0, "details": f"Error retrieving cash flow data: {str(e)}"}
    
    try:
        income_statement = company_ticker.income_stmt
        if income_statement is None or income_statement.empty:
            return {"score": 0, "max_score": 0, "details": "No income statement data available"}
    except Exception as e:
        return {"score": 0, "max_score": 0, "details": f"Error retrieving income statement data: {str(e)}"}
    
    
    # 1. Capital allocation - Check FCF to net income ratio
    # Munger values companies that convert earnings to cash

    fcf_df = safe_get_row(cash_flow, "Free Cash Flow")
    net_income_df = safe_get_row(financials, "Net Income")
    
    fcf_values = filter_valid_values(fcf_df)
    net_income_values = filter_valid_values(net_income_df)

    if fcf_values and net_income_values and len(fcf_values) == len(net_income_values):
        # Calculate FCF to Net Income ratio for each period
        fcf_to_ni_ratios = []
        for i in range(len(fcf_values)):
            if net_income_values[i] and net_income_values[i] > 0:
                fcf_to_ni_ratios.append(fcf_values[i] / net_income_values[i])
        
        if fcf_to_ni_ratios:
            avg_ratio = sum(fcf_to_ni_ratios) / len(fcf_to_ni_ratios)
            if avg_ratio > 1.1:  # FCF > net income suggests good accounting
                score += 3
                details.append(f"Excellent cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
            elif avg_ratio > 0.9:  # FCF roughly equals net income
                score += 2
                details.append(f"Good cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
            elif avg_ratio > 0.7:  # FCF somewhat lower than net income
                score += 1
                details.append(f"Moderate cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
            else:
                details.append(f"Poor cash conversion: FCF/NI ratio of only {avg_ratio:.2f}")
        else:
            details.append("Could not calculate FCF to Net Income ratios")
    else:
        details.append("Missing FCF or Net Income data")

    max_score += 2 # Increment by the *highest* possible score for this section

    
    # 2. Debt Management - Munger prefers conservative debt levels

    total_debt = safe_get_row(balance_sheet, "Total Debt")
    shareholders_equity = safe_get_row(balance_sheet, "Stockholders Equity")

    total_debt_values = filter_valid_values(total_debt)
    shareholders_equity_values = filter_valid_values(shareholders_equity)

    if total_debt_values and shareholders_equity_values and len(total_debt_values) == len(shareholders_equity_values):
        # Calculate D/E ratio for most recent period
        recent_de_ratio = total_debt_values[0] / shareholders_equity_values[0] if shareholders_equity_values[0] > 0 else float('inf')
        
        if recent_de_ratio < 0.3:  # Very low debt
            score += 3
            details.append(f"Conservative debt management: D/E ratio of {recent_de_ratio:.2f}")
        elif recent_de_ratio < 0.7:  # Moderate debt
            score += 2
            details.append(f"Prudent debt management: D/E ratio of {recent_de_ratio:.2f}")
        elif recent_de_ratio < 1.5:  # Higher but still reasonable debt
            score += 1
            details.append(f"Moderate debt level: D/E ratio of {recent_de_ratio:.2f}")
        else:
            details.append(f"High debt level: D/E ratio of {recent_de_ratio:.2f}")
    else:
        details.append("Missing debt or equity data")
    

    max_score += 3 # Increment by the *highest* possible score for this section
    
    # 3. Cash management efficiency - Munger values appropriate cash levels

    cash_and_equivalents = safe_get_row(balance_sheet, "Cash And Cash Equivalents")
    total_revenue = safe_get_row(financials, "Total Revenue")

    cash_values = filter_valid_values(cash_and_equivalents)
    revenue_values = filter_valid_values(total_revenue)

    if cash_values and revenue_values and len(cash_values) > 0 and len(revenue_values) > 0:
        # Calculate cash to revenue ratio (Munger likes 10-20% for most businesses)
        cash_to_revenue = cash_values[0] / revenue_values[0] if revenue_values[0] > 0 else 0
        
        if 0.1 <= cash_to_revenue <= 0.25:
            # Goldilocks zone - not too much, not too little
            score += 2
            details.append(f"Prudent cash management: Cash/Revenue ratio of {cash_to_revenue:.2f}")
        elif 0.05 <= cash_to_revenue < 0.1 or 0.25 < cash_to_revenue <= 0.4:
            # Reasonable but not ideal
            score += 1
            details.append(f"Acceptable cash position: Cash/Revenue ratio of {cash_to_revenue:.2f}")
        elif cash_to_revenue > 0.4:
            # Too much cash - potentially inefficient capital allocation
            details.append(f"Excess cash reserves: Cash/Revenue ratio of {cash_to_revenue:.2f}")
        else:
            # Too little cash - potentially risky
            details.append(f"Low cash reserves: Cash/Revenue ratio of {cash_to_revenue:.2f}")
    else:
        details.append("Insufficient cash or revenue data")

    max_score += 2 # Increment by the *highest* possible score for this section

    
    # 4. Insider activity - Munger values skin in the game
    
    insider_trades = company_ticker.insider_purchases

    if insider_trades.empty:
        details.append("No insider trades data available; defaulting to neutral.")
        return {"score": score, "details": "; ".join(details)}
    
    buys, sells = insider_trades["Trans"].iloc[0], insider_trades["Trans"].iloc[1]
    
    total_trades = buys + sells

    if total_trades > 0:
            buy_ratio = buys / total_trades
            if buy_ratio > 0.7:  # Strong insider buying
                score += 2
                details.append(f"Strong insider buying: {buys}/{total_trades} transactions are purchases")
            elif buy_ratio > 0.4:  # Balanced insider activity
                score += 1
                details.append(f"Balanced insider trading: {buys}/{total_trades} transactions are purchases")
            elif buy_ratio < 0.1 and sells > 5:  # Heavy selling
                score -= 1  # Penalty for excessive selling
                details.append(f"Concerning insider selling: {sells}/{total_trades} transactions are sales")
            else:
                details.append(f"Mixed insider activity: {buys}/{total_trades} transactions are purchases")
    else:
        details.append("No insider trading data available")

    max_score += 2 # Increment by the *highest* possible score for this section


    # 5. Consistency in share count - Munger prefers stable/decreasing shares

    outstanding_shares = safe_get_row(income_statement, "Basic Average Shares")
    
    if not outstanding_shares.empty:
        share_counts = filter_valid_values(outstanding_shares)

    if share_counts and len(share_counts) >= 3:
        if share_counts[0] < share_counts[-1] * 0.95:  # 5%+ reduction in shares
            score += 2
            details.append("Shareholder-friendly: Reducing share count over time")
        elif share_counts[0] < share_counts[-1] * 1.05:  # Stable share count
            score += 1
            details.append("Stable share count: Limited dilution")
        elif share_counts[0] > share_counts[-1] * 1.2:  # >20% dilution
            score -= 1  # Penalty for excessive dilution
            details.append("Concerning dilution: Share count increased significantly")
        else:
            details.append("Moderate share count increase over time")
    else:
        details.append("Insufficient share count data")

    max_score += 2 # Increment by the *highest* possible score for this section

    # Scale score to 0-10 range
    final_score = round(max(0, min(10, score * 10 / max_score)), 2)
    
    return {
        "score": final_score,
        "normalized_max_score": 10,
        "details": "; ".join(details)
    }


def analyse_predictability(ticker: str) -> dict:
    """
    Assess the predictability of the business - Munger strongly prefers businesses
    whose future operations and cashflows are relatively easy to predict.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        dict: Analysis results with score and details
    """
    years_back = 4
    score = 0
    max_score = 0
    details = []
    
    try:
        company_ticker = get_ticker(ticker)
    except Exception as e:
        return {"score": 0, "max_score": 0, "details": f"Failed to get ticker data: {str(e)}"}
    
    try:
        financials = company_ticker.financials
        if financials is None or financials.empty:
            return {"score": 0, "max_score": 0, "details": "No financial data available"}
    except Exception as e:
        return {"score": 0, "max_score": 0, "details": f"Error retrieving financial data: {str(e)}"}

    try:
        cash_flow = company_ticker.cashflow
        if cash_flow is None or cash_flow.empty:
            return {"score": 0, "max_score": 0, "details": "No cash flow data available"}
    except Exception as e:
        return {"score": 0, "max_score": 0, "details": f"Error retrieving cash flow data: {str(e)}"}
    

    # 1. Revenue Consistency - Munger values predictable revenue streams
    total_revenue = safe_get_row(financials, "Total Revenue")
    
    if not total_revenue.empty:
        revenues = filter_valid_values(total_revenue)
        if revenues and len(revenues) >= years_back:
            # Calculate year-over-year growth rates
            growth_rates = [(revenues[i] / revenues[i+1] - 1) for i in range(len(revenues)-1)]
            avg_growth = sum(growth_rates) / len(growth_rates)
            growth_volatility = sum(abs(r - avg_growth) for r in growth_rates) / len(growth_rates)
            
            if avg_growth > 0.05 and growth_volatility < 0.1:
                # Steady, consistent growth (Munger loves this)
                score += 3
                details.append(f"Highly predictable revenue: {avg_growth:.1%} avg growth with low volatility")
            elif avg_growth > 0 and growth_volatility < 0.2:
                # Positive but somewhat volatile growth
                score += 2
                details.append(f"Moderately predictable revenue: {avg_growth:.1%} avg growth with some volatility")
            elif avg_growth > 0:
                # Growing but unpredictable
                score += 1
                details.append(f"Growing but less predictable revenue: {avg_growth:.1%} avg growth with high volatility")
            else:
                details.append(f"Declining or highly unpredictable revenue: {avg_growth:.1%} avg growth")
    else:
        details.append("Insufficient revenue history for predictability analysis")

    max_score += 3 # Increment by the *highest* possible score for this section (2 + 2)
    
    # 2. Operating income Consistency - Munger values predictable earnings

    operating_income = safe_get_row(financials, "Operating Income")

    if not operating_income.empty:
        op_income = filter_valid_values(operating_income)
        if op_income and len(op_income) >= years_back:
            # Count positive operating income periods
            positive_periods = sum(1 for income in op_income if income > 0)
            
            if positive_periods == len(op_income):
                # Consistently profitable operations
                score += 3
                details.append("Highly predictable operations: Operating income positive in all periods")
            elif positive_periods >= len(op_income) * 0.8:
                # Mostly profitable operations
                score += 2
                details.append(f"Predictable operations: Operating income positive in {positive_periods}/{len(op_income)} periods")
            elif positive_periods >= len(op_income) * 0.6:
                # Somewhat profitable operations
                score += 1
                details.append(f"Somewhat predictable operations: Operating income positive in {positive_periods}/{len(op_income)} periods")
            else:
                details.append(f"Unpredictable operations: Operating income positive in only {positive_periods}/{len(op_income)} periods")
    else:
        details.append("Insufficient operating income history")

    max_score += 3 # Increment by the *highest* possible score for this section

    # 3. Margin consistency - Munger values stable margins

    operating_income = safe_get_row(financials, "Operating Income")
    total_revenue = safe_get_row(financials, "Total Revenue")

    if not operating_income.empty and not total_revenue.empty:
        operating_margins = [operating_income.iloc[i] / total_revenue.iloc[i] for i in range(len(operating_income)) if total_revenue.iloc[i] != 0]

        op_margins = filter_valid_values(operating_margins)

        if op_margins and len(op_margins) >= years_back:
            # Calculate margin volatility
            avg_margin = sum(op_margins) / len(op_margins)
            margin_volatility = sum(abs(m - avg_margin) for m in op_margins) / len(op_margins)
            
            if margin_volatility < 0.03:  # Very stable margins
                score += 2
                details.append(f"Highly predictable margins: {avg_margin:.1%} avg with minimal volatility")
            elif margin_volatility < 0.07:  # Moderately stable margins
                score += 1
                details.append(f"Moderately predictable margins: {avg_margin:.1%} avg with some volatility")
            else:
                details.append(f"Unpredictable margins: {avg_margin:.1%} avg with high volatility ({margin_volatility:.1%})")
    else:
        details.append("Insufficient margin history")

    max_score += 2 # Increment by the *highest* possible score for this section


    # 4. Cash generation reliability

    free_cash_flow = safe_get_row(cash_flow, "Free Cash Flow")

    if not free_cash_flow.empty:
        fcf_values = filter_valid_values(free_cash_flow)
        if fcf_values and len(fcf_values) >= years_back:
            # Count positive FCF periods
            positive_fcf_periods = sum(1 for fcf in fcf_values if fcf > 0)
            
            if positive_fcf_periods == len(fcf_values):
                # Consistently positive FCF
                score += 2
                details.append("Highly predictable cash generation: Positive FCF in all periods")
            elif positive_fcf_periods >= len(fcf_values) * 0.8:
                # Mostly positive FCF
                score += 1
                details.append(f"Predictable cash generation: Positive FCF in {positive_fcf_periods}/{len(fcf_values)} periods")
            else:
                details.append(f"Unpredictable cash generation: Positive FCF in only {positive_fcf_periods}/{len(fcf_values)} periods")
    else:
        details.append("Insufficient free cash flow history")

    max_score += 2 # Increment by the *highest* possible score for this section

    # Scale score to 0-10 range
    # Maximum possible raw score would be 10 (3+3+2+2)
    final_score = round(max(0, min(10, score * 10 / max_score)), 2)
    
    return {
        "score": final_score,
        "normalized_max_score": 10,
        "details": "; ".join(details)
    }


def calculate_munger_valuation(ticker: str) -> dict:
    """
    Calculate intrinsic value using Munger's approach:
    - Focus on owner earnings (approximated by FCF)
    - Simple multiple on normalized earnings
    - Prefer paying a fair price for a wonderful business
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        dict: Valuation analysis with score and details
    """
    score = 0
    max_score = 0 # Initialize max_score
    details = []
    
    try:
        company_ticker = get_ticker(ticker)
    except Exception as e:
        return {"score": 0, "max_score": 0, "details": f"Failed to get ticker data: {str(e)}"}
    
    try:
        cash_flow = company_ticker.cashflow
        if cash_flow is None or cash_flow.empty:
            return {"score": 0, "max_score": 0, "details": "No cash flow data available"}
    except Exception as e:
        return {"score": 0, "max_score": 0, "details": f"Error retrieving cash flow data: {str(e)}"}
    
    market_cap = company_ticker.info.get("marketCap", None)
    if market_cap is None:
        return {"score": 0, "max_score": 0, "details": "Market cap data not available"}
    
    free_cash_flow = safe_get_row(cash_flow, "Free Cash Flow")

    if not free_cash_flow.empty:
        fcf_values = filter_valid_values(free_cash_flow)
        if not fcf_values or len(fcf_values) < 3:
            return {"score": 0, "max_score": 0, "details": "Insufficient free cash flow data for valuation"}
        
        # 1. Normalize earnings by taking average of last 3-5 years
        # (Munger prefers to normalize earnings to avoid over/under-valuation based on cyclical factors)
        normalized_fcf = sum(fcf_values[:min(5, len(fcf_values))]) / min(5, len(fcf_values))
        
        if normalized_fcf <= 0:
            return {
                "score": 0,
                "details": f"Negative or zero normalized FCF ({normalized_fcf}), cannot value",
                "intrinsic_value": None
            }
        
        # 2. Calculate FCF yield (inverse of P/FCF multiple)
        fcf_yield = normalized_fcf / market_cap
        
        # 3. Apply Munger's FCF multiple based on business quality
        # Munger would pay higher multiples for wonderful businesses
        # Let's use a sliding scale where higher FCF yields are more attractive
        if fcf_yield > 0.08:  # >8% FCF yield (P/FCF < 12.5x)
            score += 4
            details.append(f"Excellent value: {fcf_yield:.1%} FCF yield")
        elif fcf_yield > 0.05:  # >5% FCF yield (P/FCF < 20x)
            score += 3
            details.append(f"Good value: {fcf_yield:.1%} FCF yield")
        elif fcf_yield > 0.03:  # >3% FCF yield (P/FCF < 33x)
            score += 1
            details.append(f"Fair value: {fcf_yield:.1%} FCF yield")
        else:
            details.append(f"Expensive: Only {fcf_yield:.1%} FCF yield")

        max_score += 4
        
        # 4. Calculate simple intrinsic value range
        # Munger tends to use straightforward valuations, avoiding complex DCF models
        conservative_value = normalized_fcf * 10  # 10x FCF = 10% yield
        reasonable_value = normalized_fcf * 15    # 15x FCF ≈ 6.7% yield
        optimistic_value = normalized_fcf * 20    # 20x FCF = 5% yield
        
        # 5. Calculate margins of safety
        current_to_reasonable = (reasonable_value - market_cap) / market_cap
        
        if current_to_reasonable > 0.3:  # >30% upside
            score += 3
            details.append(f"Large margin of safety: {current_to_reasonable:.1%} upside to reasonable value")
        elif current_to_reasonable > 0.1:  # >10% upside
            score += 2
            details.append(f"Moderate margin of safety: {current_to_reasonable:.1%} upside to reasonable value")
        elif current_to_reasonable > -0.1:  # Within 10% of reasonable value
            score += 1
            details.append(f"Fair price: Within 10% of reasonable value ({current_to_reasonable:.1%})")
        else:
            details.append(f"Expensive: {-current_to_reasonable:.1%} premium to reasonable value")


        max_score += 3
        
        # 6. Check earnings trajectory for additional context
        # Munger likes growing owner earnings
        if len(fcf_values) >= 3:
            recent_avg = sum(fcf_values[:3]) / 3
            older_avg = sum(fcf_values[-3:]) / 3 if len(fcf_values) >= 6 else fcf_values[-1]
            
            if recent_avg > older_avg * 1.2:  # >20% growth in FCF
                score += 3
                details.append("Growing FCF trend adds to intrinsic value")
            elif recent_avg > older_avg:
                score += 2
                details.append("Stable to growing FCF supports valuation")
            else:
                details.append("Declining FCF trend is concerning")


        max_score += 3
    
    # Scale score to 0-10 range
    # Maximum possible raw score would be 10 (4+3+3)
    final_score = round(min(10, score * 10 / max_score), 2) 
    
    return {
        "score": final_score,
        "normalized_max_score": 10,
        "details": "; ".join(details),
        "intrinsic_value_range": {
            "conservative": f"${conservative_value:,.2f}",
            "reasonable": f"${reasonable_value:,.2f}",
            "optimistic": f"${optimistic_value:,.2f}"
        },
        "fcf_yield": f"{fcf_yield:.2%}",
        "normalized_fcf": f"${normalized_fcf:,.2f}"
    }


def analyse_news_sentiment(ticker: str, number_of_articles: int = 10) -> str:
    """
    Analyze recent news sentiment for the company.
    
    Args:
        ticker (str): Stock ticker symbol
        number_of_articles (int, optional): Number of articles to analyze
        
    Returns:
        str: Sentiment analysis result
    """

    
    try:
        sentiment_score = get_news_sentiment_scores(ticker, number_of_articles)

        result = {
            "score": sentiment_score,
            "normalized_max_score": 10,
            "details": f"News sentiment score: {sentiment_score} (range: -10 to 10)"
        }

        return result
    except Exception as e:
        return f"Could not analyze news sentiment: {str(e)}"


def analyse_charlie_munger_valuation(ticker: str) -> dict:
    """
    Analyzes stocks using Charlie Munger's investing principles and mental models.
    Focuses on moat strength, management quality, predictability, and valuation.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        dict: Complete Munger analysis with signal, score, and detailed components
    """
       
    try:
        moat_analysis = analyse_moat_strength(ticker)
    except Exception as e:
        moat_analysis = {
            "score": 0, 
            "max_score": 0,
            "details": f"Error in moat strength analysis: {str(e)}"
        }
    
    try:
        management_analysis = analyse_management_quality(ticker)
    except Exception as e:
        management_analysis = {
            "score": 0, 
            "max_score": 0, # Initialize max_score
            "details": f"Error in management quality analysis: {str(e)}"
        }
    
    try:
        predictability_analysis = analyse_predictability(ticker)
    except Exception as e:
        predictability_analysis = {
            "score": 0, 
            "max_score": 0, # Initialize max_score
            "details": f"Error in predictability analysis: {str(e)}"
        }
    
    try:
        valuation_analysis = calculate_munger_valuation(ticker)
    except Exception as e:
        valuation_analysis = {
            "score": 0, 
            "max_score": 0, # Initialize max_score
            "details": f"Error in valuation analysis: {str(e)}"
        }
    
    try:
        news_sentiment = analyse_news_sentiment(ticker)
    except Exception as e:
        news_sentiment = f"Error in news sentiment analysis: {str(e)}"
    
    # Calculate total score
    total_score = (
        moat_analysis.get("score", 0) + 
        management_analysis.get("score", 0) + 
        predictability_analysis.get("score", 0) + 
        valuation_analysis.get("score", 0) + 
        news_sentiment.get("score", 0)
    )

    max_possible_score = (
        moat_analysis.get("normalized_max_score", 10) +
        management_analysis.get("normalized_max_score", 10) +
        predictability_analysis.get("normalized_max_score", 10) +
        valuation_analysis.get("normalized_max_score", 10) +
        news_sentiment.get("normalized_max_score", 10)
    )
    
    # Generate signal based on score
    if total_score >= 0.7 * max_possible_score:
        signal = "bullish"
    elif total_score <= 0.3 * max_possible_score:
        signal = "bearish"
    else:
        signal = "neutral"
    
    # Combine all analysis results
    munger_analysis = {
        "signal": signal,
        "score": round(total_score, 2),
        "max_score": round(max_possible_score, 2),
        "moat_analysis": moat_analysis,
        "management_analysis": management_analysis,
        "predictability_analysis": predictability_analysis,
        "valuation_analysis": valuation_analysis,
        "news_sentiment": news_sentiment
    }

    return munger_analysis


if __name__ == "__main__":
    ticker = "NVDA"
    print(analyse_charlie_munger_valuation(ticker))