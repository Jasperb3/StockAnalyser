from stock_analyser.tools.tool_utils.metrics import compute_financial_metrics
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


def analyse_moat_strength(metrics: dict, ticker: str, company_ticker=None, financials=None):
    """
    Analyze the company's economic moat strength based on Munger's principles.
    
    Args:
        metrics (dict): Financial metrics dictionary
        ticker (str): Stock ticker symbol
        company_ticker (yf.Ticker, optional): Ticker object to reuse
        financials (pd.DataFrame, optional): Financial data to reuse
        
    Returns:
        dict: Analysis results with score and details
    """
    score = 0
    max_score = 0 # Initialize max_score
    details = []
    
    # Reuse ticker object if provided, otherwise get a new one
    if company_ticker is None:
        try:
            company_ticker = get_ticker(ticker)
        except Exception as e:
            return {"score": 0, "max_score": 0, "details": f"Failed to get ticker data: {str(e)}"}
    
    # Reuse financials if provided, otherwise get from ticker
    if financials is None:
        try:
            financials = company_ticker.financials
            if financials is None or financials.empty:
                return {"score": 0, "max_score": 0, "details": "No financial data available"}
        except Exception as e:
            return {"score": 0, "max_score": 0, "details": f"Error retrieving financial data: {str(e)}"}
    
    if not metrics:
        return {"score": 0, "max_score": 0, "details": "Insufficient data for moat strength analysis"}
    
    # 1. Return on Equity (ROE) - Munger looks for consistently high ROE
    roe = metrics.get("return_on_equity")
    if roe is not None and not np.isnan(roe):
        if roe > 0.20:  # 20%+ ROE
            score += 3
            details.append(f"Exceptional ROE of {roe:.1%}")
        elif roe > 0.15:  # 15-20% ROE
            score += 2
            details.append(f"Strong ROE of {roe:.1%}")
        elif roe > 0.10:  # 10-15% ROE
            score += 1
            details.append(f"Good ROE of {roe:.1%}")
        else:
            details.append(f"Moderate ROE of {roe:.1%}")
    else:
        details.append("ROE data not available")

    max_score += 3 # Increment by the *highest* possible score for this section
    
    # 2. Gross Margin - High margins often indicate pricing power
    gross_margin = metrics.get("gross_margin")
    if gross_margin is not None and not np.isnan(gross_margin):
        if gross_margin > 0.50:  # 50%+ gross margin
            score += 2
            details.append(f"High gross margin of {gross_margin:.1%} indicates pricing power")
        elif gross_margin > 0.30:  # 30-50% gross margin
            score += 1
            details.append(f"Solid gross margin of {gross_margin:.1%}")
        else:
            details.append(f"Lower gross margin of {gross_margin:.1%}")
    else:
        details.append("Gross margin data not available")

    max_score += 2 # Increment by the *highest* possible score for this section
    
    # 3. Operating Margin Consistency - Munger values stable, high operating margins
    operating_income_df = safe_get_row(financials, "Operating Income", ["Operating Profit"])
    revenue_df = safe_get_row(financials, "Total Revenue", ["Revenue", "Revenues"])
    
    if operating_income_df is not None and revenue_df is not None:
        op_incomes = filter_valid_values(operating_income_df)
        revenues = filter_valid_values(revenue_df)
        
        if len(op_incomes) >= 2 and len(revenues) >= 2 and len(op_incomes) == len(revenues):
            # Calculate operating margins for all periods
            op_margins = []
            for i in range(len(op_incomes)):
                if revenues[i] > 0:  # Avoid division by zero
                    op_margins.append(op_incomes[i] / revenues[i])
            
            if op_margins:
                # Check if margins are consistently high
                high_margin_count = sum(1 for m in op_margins if m > 0.15)
                if high_margin_count == len(op_margins) and len(op_margins) >= 3:
                    score += 3
                    details.append(f"Consistently high operating margins across {len(op_margins)} periods")
                elif high_margin_count >= len(op_margins) * 0.75:
                    score += 2
                    details.append(f"Strong operating margins in most periods ({high_margin_count}/{len(op_margins)})")
                elif high_margin_count > 0:
                    score += 1
                    details.append(f"Some periods with strong operating margins ({high_margin_count}/{len(op_margins)})")
                else:
                    details.append("No periods with high operating margins")
                
                # Check margin stability (low variance is good)
                if len(op_margins) >= 3:
                    margin_variance = np.var(op_margins)
                    if margin_variance < 0.0025:  # Low variance threshold
                        score += 1
                        details.append("Very stable operating margins (low variance)")
                    elif margin_variance < 0.01:
                        details.append("Moderately stable operating margins")
                    else:
                        details.append("Volatile operating margins")
            else:
                details.append("Could not calculate operating margins (division by zero)")
        else:
            details.append("Insufficient data for operating margin analysis")
    else:
        details.append("Operating income or revenue data not available")

    max_score += 4 # Increment by the *highest* possible score for this section (3 + 1)
    
    # 4. Capital Efficiency - Munger likes businesses that don't require much capital
    asset_turnover = metrics.get("asset_turnover")
    if asset_turnover is not None and not np.isnan(asset_turnover):
        if asset_turnover > 1.5:
            score += 2
            details.append(f"Excellent capital efficiency (asset turnover: {asset_turnover:.2f})")
        elif asset_turnover > 0.8:
            score += 1
            details.append(f"Good capital efficiency (asset turnover: {asset_turnover:.2f})")
        else:
            details.append(f"Lower capital efficiency (asset turnover: {asset_turnover:.2f})")
    else:
        details.append("Asset turnover data not available")

    max_score += 2 # Increment by the *highest* possible score for this section
    
    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def analyse_management_quality(metrics: dict, ticker: str, company_ticker=None, balance_sheet=None, cash_flow=None) -> dict:
    """
    Analyze the quality of company management based on Munger's principles.
    
    Args:
        metrics (dict): Financial metrics dictionary
        ticker (str): Stock ticker symbol
        company_ticker (yf.Ticker, optional): Ticker object to reuse
        balance_sheet (pd.DataFrame, optional): Balance sheet data to reuse
        cash_flow (pd.DataFrame, optional): Cash flow data to reuse
        
    Returns:
        dict: Analysis results with score and details
    """
    score = 0
    max_score = 0 # Initialize max_score
    details = []
    
    # Reuse ticker object if provided, otherwise get a new one
    if company_ticker is None:
        try:
            company_ticker = get_ticker(ticker)
        except Exception as e:
            return {"score": 0, "max_score": 0, "details": f"Failed to get ticker data: {str(e)}"}
    
    # Reuse balance_sheet if provided, otherwise get from ticker
    if balance_sheet is None:
        try:
            balance_sheet = company_ticker.balance_sheet
            if balance_sheet is None or balance_sheet.empty:
                return {"score": 0, "max_score": 0, "details": "No balance sheet data available"}
        except Exception as e:
            return {"score": 0, "max_score": 0, "details": f"Error retrieving balance sheet data: {str(e)}"}
            
    # Reuse cash_flow if provided, otherwise get from ticker
    if cash_flow is None:
        try:
            cash_flow = company_ticker.cashflow
            if cash_flow is None or cash_flow.empty:
                return {"score": 0, "max_score": 0, "details": "No cash flow data available"}
        except Exception as e:
            return {"score": 0, "max_score": 0, "details": f"Error retrieving cash flow data: {str(e)}"}
    
    if not metrics:
        return {"score": 0, "max_score": 0, "details": "Insufficient data for management quality analysis"}
    
    # 1. Capital Allocation - Check for share repurchases when appropriate
    shares_df = safe_get_row(balance_sheet, "Share Issued", 
                            ["Common Stock Shares Outstanding", "Ordinary Shares Number"])
    
    if shares_df is not None:
        shares = filter_valid_values(shares_df)
        if len(shares) >= 2:
            # Note: yfinance data is typically in reverse chronological order (newest first)
            latest = shares[0]
            earliest = None

            # Find a valid (non-zero) earliest share count
            for i in range(len(shares) - 1, -1, -1):
                if shares[i] != 0:
                    earliest = shares[i]
                    break

            if earliest is not None:
                if latest < earliest:
                    # Check if the company has a good ROE while buying back shares
                    roe = metrics.get("return_on_equity")
                    if roe is not None and not np.isnan(roe) and roe > 0.15:
                        score += 3
                        details.append(f"Intelligent capital allocation: share count reduced with high ROE ({roe:.1%})")
                    else:
                        score += 2
                        details.append("Share count reduced over time (potential buybacks)")
                else:
                    details.append("Share count has increased over time")
            elif len(shares) > 1:
                details.append("Cannot calculate share change due to zero or invalid values.")
            else:
                details.append("Insufficient share count data for trend analysis")
        else:
            details.append("Insufficient share count data for trend analysis")
    else:
        details.append("Share count data not available")

    max_score += 3 # Increment by the *highest* possible score for this section
    
    # 2. Debt Management - Munger prefers conservative debt levels
    debt_to_equity = metrics.get("debt_to_equity")
    if debt_to_equity is not None and not np.isnan(debt_to_equity):
        if debt_to_equity < 0.3:
            score += 2
            details.append(f"Very conservative debt management (D/E ratio: {debt_to_equity:.2f})")
        elif debt_to_equity < 0.7:
            score += 1
            details.append(f"Prudent debt management (D/E ratio: {debt_to_equity:.2f})")
        elif debt_to_equity < 1.5:
            details.append(f"Moderate debt levels (D/E ratio: {debt_to_equity:.2f})")
        else:
            details.append(f"Higher debt levels (D/E ratio: {debt_to_equity:.2f})")
    else:
        details.append("Debt to equity data not available")

    max_score += 2 # Increment by the *highest* possible score for this section
    
    # 3. Free Cash Flow Generation - Munger values consistent FCF
    fcf_df = safe_get_row(cash_flow, "Free Cash Flow")
    
    if fcf_df is not None:
        fcf_values = filter_valid_values(fcf_df)
        if len(fcf_values) >= 3:
            positive_fcf_count = sum(1 for fcf in fcf_values if fcf > 0)
            if positive_fcf_count == len(fcf_values):
                score += 2
                details.append(f"Consistent positive free cash flow across all {len(fcf_values)} periods")
            elif positive_fcf_count >= len(fcf_values) * 0.75:
                score += 1
                details.append(f"Mostly positive free cash flow ({positive_fcf_count}/{len(fcf_values)} periods)")
            else:
                details.append(f"Inconsistent free cash flow (positive in {positive_fcf_count}/{len(fcf_values)} periods)")
        else:
            details.append("Insufficient free cash flow data for trend analysis")
    else:
        details.append("Free cash flow data not available")

    max_score += 2 # Increment by the *highest* possible score for this section
    
    # 4. Capital Expenditure Efficiency
    capex = metrics.get("capex")
    depreciation = metrics.get("depreciation_and_amortization")
    
    if capex is not None and not np.isnan(capex) and depreciation is not None and not np.isnan(depreciation) and depreciation != 0:
        capex_to_depreciation = capex / depreciation
        if capex_to_depreciation < 1.2:
            score += 2
            details.append(f"Efficient capital expenditure (CapEx/Depreciation: {capex_to_depreciation:.2f})")
        elif capex_to_depreciation < 1.5:
            score += 1
            details.append(f"Reasonable capital expenditure (CapEx/Depreciation: {capex_to_depreciation:.2f})")
        else:
            details.append(f"Higher capital expenditure (CapEx/Depreciation: {capex_to_depreciation:.2f})")
    else:
        details.append("Capital expenditure or depreciation data not available")

    max_score += 2 # Increment by the *highest* possible score for this section
    
    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def analyse_predictability(metrics: dict, ticker: str, company_ticker=None, financials=None) -> dict:
    """
    Analyze the predictability and consistency of the business based on Munger's principles.
    
    Args:
        metrics (dict): Financial metrics dictionary
        ticker (str): Stock ticker symbol
        company_ticker (yf.Ticker, optional): Ticker object to reuse
        financials (pd.DataFrame, optional): Financial data to reuse
        
    Returns:
        dict: Analysis results with score and details
    """
    score = 0
    max_score = 0 # Initialize max_score
    details = []
    
    # Reuse ticker object if provided, otherwise get a new one
    if company_ticker is None:
        try:
            company_ticker = get_ticker(ticker)
        except Exception as e:
            return {"score": 0, "max_score": 0, "details": f"Failed to get ticker data: {str(e)}"}
    
    # Reuse financials if provided, otherwise get from ticker
    if financials is None:
        try:
            financials = company_ticker.financials
            if financials is None or financials.empty:
                return {"score": 0, "max_score": 0, "details": "No financial data available"}
        except Exception as e:
            return {"score": 0, "max_score": 0, "details": f"Error retrieving financial data: {str(e)}"}
    
    if not metrics:
        return {"score": 0, "max_score": 0, "details": "Insufficient data for predictability analysis"}
    
    # 1. Revenue Consistency - Munger values predictable revenue streams
    revenue_df = safe_get_row(financials, "Total Revenue", ["Revenue", "Revenues"])
    
    if revenue_df is not None:
        revenues = filter_valid_values(revenue_df)
        if len(revenues) >= 3:
            # Check if revenue is consistently growing
            # is_growing = True
            # for i in range(len(revenues) - 1):
            #     if revenues[i] <= revenues[i+1]:  # Remember yfinance data is newest first
            #         is_growing = False
            #         break
            
            # if is_growing:
            #     score += 2
            #     details.append(f"Consistent revenue growth across {len(revenues)} periods")

            # MODIFIED SECTION:
            latest = revenues[0]
            earliest = None

            for i in range(len(revenues) - 1, -1, -1):
                if revenues[i] != 0:
                    earliest = revenues[i]
                    break

            if earliest is not None:
                if latest > earliest:
                    score += 2
                    details.append(f"Revenue grew from {earliest} to {latest}")
                else:
                    details.append(f"Revenue did not grow from {earliest} to {latest}")
            elif len(revenues) > 1:
                details.append("Cannot calculate revenue growth due to zero or invalid values.")
            else:
                details.append("Insufficient data for revenue growth analysis")
            # END MODIFIED SECTION
            
            # Check revenue volatility
            if len(revenues) >= 4:
                # Calculate year-over-year growth rates
                growth_rates = []
                for i in range(len(revenues) - 1):
                    if revenues[i+1] > 0:  # Avoid division by zero
                        growth_rates.append((revenues[i] - revenues[i+1]) / revenues[i+1])
                
                if growth_rates:
                    growth_variance = np.var(growth_rates)
                    if growth_variance < 0.01:
                        score += 2
                        details.append("Very stable revenue growth (low variance)")
                    elif growth_variance < 0.04:
                        score += 1
                        details.append("Moderately stable revenue growth")
                    else:
                        details.append("Volatile revenue growth")
                else:
                    details.append("Could not calculate revenue growth rates")
            else:
                details.append("Insufficient data for revenue volatility analysis")
        else:
            details.append("Insufficient revenue data for trend analysis")
    else:
        details.append("Revenue data not available")

    max_score += 4 # Increment by the *highest* possible score for this section (2 + 2)
    
    # 2. Earnings Consistency - Munger values predictable earnings
    net_income_df = safe_get_row(financials, "Net Income", 
                                ["Net Income Common Stockholders", "Net Profit"])
    
    if net_income_df is not None:
        net_incomes = filter_valid_values(net_income_df)
        if len(net_incomes) >= 3:
            # Check for consistent profitability
            profitable_periods = sum(1 for ni in net_incomes if ni > 0)
            if profitable_periods == len(net_incomes):
                score += 2
                details.append(f"Consistently profitable across all {len(net_incomes)} periods")
            elif profitable_periods >= len(net_incomes) * 0.75:
                score += 1
                details.append(f"Profitable in most periods ({profitable_periods}/{len(net_incomes)})")
            else:
                details.append(f"Inconsistent profitability ({profitable_periods}/{len(net_incomes)} periods)")
            
            # Check earnings volatility
            if len(net_incomes) >= 4 and all(ni > 0 for ni in net_incomes):
                # Calculate year-over-year growth rates
                growth_rates = []
                for i in range(len(net_incomes) - 1):
                    if net_incomes[i+1] > 0:  # Avoid division by zero
                        growth_rates.append((net_incomes[i] - net_incomes[i+1]) / net_incomes[i+1])
                
                if growth_rates:
                    growth_variance = np.var(growth_rates)
                    if growth_variance < 0.04:
                        score += 2
                        details.append("Very stable earnings growth (low variance)")
                    elif growth_variance < 0.09:
                        score += 1
                        details.append("Moderately stable earnings growth")
                    else:
                        details.append("Volatile earnings growth")
                else:
                    details.append("Could not calculate earnings growth rates")
            else:
                details.append("Insufficient data for earnings volatility analysis")
        else:
            details.append("Insufficient earnings data for trend analysis")
    else:
        details.append("Net income data not available")

    max_score += 4 # Increment by the *highest* possible score for this section (2 + 2)
    
    # 3. Business Model Simplicity - Munger prefers simple, understandable businesses
    # This is more qualitative, but we can use sector/industry as a proxy
    info = company_ticker.info
    sector = info.get("sector", "Unknown")
    industry = info.get("industry", "Unknown")
    
    simple_sectors = [
        "Consumer Defensive", "Consumer Staples", "Consumer Discretionary",
        "Utilities", "Real Estate", "Industrials"
    ]
    
    complex_sectors = [
        "Financial Services", "Healthcare", "Biotechnology", "Technology"
    ]
    
    if any(s.lower() in sector.lower() for s in simple_sectors):
        score += 1
        details.append(f"Potentially simple business model in {sector} sector")
    elif any(s.lower() in sector.lower() for s in complex_sectors):
        details.append(f"Potentially complex business model in {sector} sector")

    max_score += 1 # Increment by the *highest* possible score for this section
    
    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def calculate_munger_valuation(metrics: dict, ticker: str, company_ticker=None) -> dict:
    """
    Calculate a valuation based on Munger's principles of paying a fair price for quality.
    
    Args:
        metrics (dict): Financial metrics dictionary
        ticker (str): Stock ticker symbol
        company_ticker (yf.Ticker, optional): Ticker object to reuse
        
    Returns:
        dict: Valuation analysis with score and details
    """
    score = 0
    max_score = 0 # Initialize max_score
    details = []
    
    # Reuse ticker object if provided, otherwise get a new one
    if company_ticker is None:
        try:
            company_ticker = get_ticker(ticker)
        except Exception as e:
            return {"score": 0, "max_score": 0, "details": f"Failed to get ticker data: {str(e)}"}
    
    if not metrics:
        return {"score": 0, "max_score": 0, "details": "Insufficient data for valuation analysis"}
    
    # 1. P/E Ratio - Munger is willing to pay more for quality, but still values reasonable P/E
    pe_ratio = metrics.get("price_to_earnings_ratio")
    if pe_ratio is not None and not np.isnan(pe_ratio) and pe_ratio > 0:
        # Adjust PE thresholds based on ROE (higher quality = higher acceptable PE)
        roe = metrics.get("return_on_equity")
        if roe is not None and not np.isnan(roe) and roe > 0:
            # Quality-adjusted PE threshold
            quality_pe_threshold = 15 + (roe * 100 - 15) * 0.5  # e.g., 20% ROE -> PE threshold of 17.5
            
            if pe_ratio < quality_pe_threshold * 0.8:
                score += 3
                details.append(f"Very attractive P/E of {pe_ratio:.1f} relative to quality (ROE: {roe:.1%})")
            elif pe_ratio < quality_pe_threshold:
                score += 2
                details.append(f"Reasonable P/E of {pe_ratio:.1f} relative to quality (ROE: {roe:.1%})")
            elif pe_ratio < quality_pe_threshold * 1.5:
                score += 1
                details.append(f"Acceptable P/E of {pe_ratio:.1f} relative to quality (ROE: {roe:.1%})")
            else:
                details.append(f"Higher P/E of {pe_ratio:.1f} relative to quality (ROE: {roe:.1%})")
        else:
            # Standard PE thresholds if ROE is not available
            if pe_ratio < 15:
                score += 2
                details.append(f"Attractive P/E ratio of {pe_ratio:.1f}")
            elif pe_ratio < 20:
                score += 1
                details.append(f"Reasonable P/E ratio of {pe_ratio:.1f}")
            else:
                details.append(f"Higher P/E ratio of {pe_ratio:.1f}")
    else:
        details.append("P/E ratio data not available or negative")

    max_score += 3 # Increment by the *highest* possible score for this section
    
    # 2. Price to Free Cash Flow - Munger values companies generating strong FCF
    fcf = metrics.get("free_cash_flow_per_share")
    if fcf is not None and not np.isnan(fcf) and fcf > 0:
        # Calculate P/FCF
        price = company_ticker.info.get("currentPrice")
        if price is not None and not np.isnan(price):
            p_fcf = price / fcf
            if p_fcf < 15:
                score += 2
                details.append(f"Attractive price to FCF ratio of {p_fcf:.1f}")
            elif p_fcf < 20:
                score += 1
                details.append(f"Reasonable price to FCF ratio of {p_fcf:.1f}")
            else:
                details.append(f"Higher price to FCF ratio of {p_fcf:.1f}")
        else:
            details.append("Current price data not available")
    else:
        details.append("Free cash flow per share data not available or negative")

    max_score += 2 # Increment by the *highest* possible score for this section
    
    # 3. Price to Book Value - Munger considers this but with less emphasis than earnings power
    pb_ratio = metrics.get("price_to_book_ratio")
    if pb_ratio is not None and not np.isnan(pb_ratio) and pb_ratio > 0:
        # Adjust PB thresholds based on ROE (higher ROE justifies higher PB)
        roe = metrics.get("return_on_equity")
        if roe is not None and not np.isnan(roe) and roe > 0:
            justified_pb = roe * 2  # Simple approximation: justified P/B ≈ 2 × ROE
            
            if pb_ratio < justified_pb * 0.7:
                score += 2
                details.append(f"Very attractive P/B of {pb_ratio:.1f} relative to ROE of {roe:.1%}")
            elif pb_ratio < justified_pb:
                score += 1
                details.append(f"Reasonable P/B of {pb_ratio:.1f} relative to ROE of {roe:.1%}")
            else:
                details.append(f"Higher P/B of {pb_ratio:.1f} relative to ROE of {roe:.1%}")
        else:
            # Standard PB thresholds if ROE is not available
            if pb_ratio < 3:
                score += 1
                details.append(f"Attractive P/B ratio of {pb_ratio:.1f}")
            else:
                details.append(f"Higher P/B ratio of {pb_ratio:.1f}")
    else:
        details.append("Price to book ratio data not available or negative")

    max_score += 2 # Increment by the *highest* possible score for this section
    
    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


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
        return f"News sentiment score: {sentiment_score:.1f} (range: -100 to 100)"
    except Exception as e:
        return f"Could not analyze news sentiment: {str(e)}"


def analyse_charlie_munger_valuation(ticker: str) -> dict:
    """
    Perform a comprehensive analysis based on Charlie Munger's investment principles.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        dict: Complete Munger analysis with signal, score, and detailed components
    """
    try:
        # Get ticker data once and reuse
        company_ticker = get_ticker(ticker)
        financials = company_ticker.financials
        balance_sheet = company_ticker.balance_sheet
        cash_flow = company_ticker.cashflow
        
        # Get metrics once
        metrics = compute_financial_metrics(ticker)
    except Exception as e:
        return {
            "signal": "neutral",
            "score": 0,
            "max_score": 0, # Initialize max_score
            "error": f"Failed to compute financial metrics: {str(e)}"
        }
    
    try:
        moat_analysis = analyse_moat_strength(metrics, ticker, company_ticker, financials)
    except Exception as e:
        moat_analysis = {
            "score": 0, 
            "max_score": 0, # Initialize max_score
            "details": f"Error in moat strength analysis: {str(e)}"
        }
    
    try:
        management_analysis = analyse_management_quality(metrics, ticker, company_ticker, balance_sheet, cash_flow)
    except Exception as e:
        management_analysis = {
            "score": 0, 
            "max_score": 0, # Initialize max_score
            "details": f"Error in management quality analysis: {str(e)}"
        }
    
    try:
        predictability_analysis = analyse_predictability(metrics, ticker, company_ticker, financials)
    except Exception as e:
        predictability_analysis = {
            "score": 0, 
            "max_score": 0, # Initialize max_score
            "details": f"Error in predictability analysis: {str(e)}"
        }
    
    try:
        valuation_analysis = calculate_munger_valuation(metrics, ticker, company_ticker)
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
        valuation_analysis.get("score", 0)
    )
    max_possible_score = (
        moat_analysis.get("max_score", 0) +
        management_analysis.get("max_score", 0) +
        predictability_analysis.get("max_score", 0) +
        valuation_analysis.get("max_score", 0)
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
        "score": total_score,
        "max_score": max_possible_score,
        "moat_analysis": moat_analysis,
        "management_analysis": management_analysis,
        "predictability_analysis": predictability_analysis,
        "valuation_analysis": valuation_analysis,
        "news_sentiment": news_sentiment
    }

    return munger_analysis


if __name__ == "__main__":
    ticker = "FGBI"
    metrics = compute_financial_metrics(ticker)
    print(analyse_charlie_munger_valuation(ticker))