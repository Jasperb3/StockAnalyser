from stock_analyser.tools.tool_utils.metrics import compute_financial_metrics
import yfinance as yf
import math
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


def calculate_cathie_wood_analysis_data(ticker: str, growth_rate: float = 0.20, 
                                       discount_rate: float = 0.15, 
                                       terminal_multiple: float = 25, 
                                       projection_years: int = 5):
    """
    Analyzes stocks using Cathie Wood's innovation-focused investing principles.
    
    Args:
        ticker (str): Stock ticker symbol
        growth_rate (float): Annual growth rate for projections (default 20%)
        discount_rate (float): Discount rate for present value calculations (default 15%)
        terminal_multiple (float): Multiple for terminal value calculation (default 25x)
        projection_years (int): Number of years to project cash flows (default 5)
        
    Returns:
        dict: Complete analysis with signal, score, and detailed components
    """
    try:
        # Get ticker data once and reuse
        company_ticker = get_ticker(ticker)
        
        # Get metrics once
        metrics = compute_financial_metrics(ticker)
    except Exception as e:
        return {
            "signal": "neutral",
            "score": 0,
            "max_score": 15,
            "error": f"Failed to compute financial metrics: {str(e)}"
        }

    try:
        disruptive_analysis = analyse_disruptive_potential(metrics, ticker, company_ticker)
    except Exception as e:
        disruptive_analysis = {
            "score": 0, 
            "details": f"Error in disruptive potential analysis: {str(e)}"
        }

    try:
        innovation_analysis = analyse_innovation_growth(metrics, ticker, company_ticker)
    except Exception as e:
        innovation_analysis = {
            "score": 0, 
            "details": f"Error in innovation growth analysis: {str(e)}"
        }

    try:
        valuation_analysis = analyse_cathie_wood_valuation(metrics, ticker, growth_rate, 
                                                         discount_rate, terminal_multiple, 
                                                         projection_years, company_ticker)
    except Exception as e:
        valuation_analysis = {
            "score": 0, 
            "details": f"Error in valuation analysis: {str(e)}"
        }

    # Combine partial scores
    total_score = disruptive_analysis.get("score", 0) + innovation_analysis.get("score", 0) + valuation_analysis.get("score", 0)
    max_possible_score = 15
    
    # Generate signal based on score
    if total_score >= 0.7 * max_possible_score:
        signal = "bullish"
    elif total_score <= 0.3 * max_possible_score:
        signal = "bearish"
    else:
        signal = "neutral"
    
    analysis_data = {
        "signal": signal,
        "score": total_score,
        "max_score": max_possible_score,
        "disruptive_analysis": disruptive_analysis,
        "innovation_analysis": innovation_analysis,
        "valuation_analysis": valuation_analysis
    }

    return analysis_data


def analyse_disruptive_potential(metrics: dict, ticker: str, company_ticker=None):
    """
    Analyze the company's potential for disruption and innovation.
    
    Args:
        metrics (dict): Financial metrics dictionary
        ticker (str): Stock ticker symbol
        company_ticker (yf.Ticker, optional): Ticker object to reuse
        
    Returns:
        dict: Analysis results with score and details
    """
    score = 0
    details = []
    
    # Reuse ticker object if provided, otherwise get a new one
    if company_ticker is None:
        try:
            company_ticker = get_ticker(ticker)
        except Exception as e:
            return {"score": 0, "details": f"Failed to get ticker data: {str(e)}"}
    
    if not metrics:
        return {"score": 0, "details": "Insufficient data for disruptive potential analysis"}
    
    # Get company info and sector
    info = company_ticker.info
    sector = info.get("sector", "Unknown")
    industry = info.get("industry", "Unknown")
    
    # 1. Check if company is in a disruptive/innovative sector
    disruptive_sectors = [
        "Technology", "Information Technology", "Healthcare", "Biotechnology", 
        "Renewable Energy", "Electric Vehicles", "Artificial Intelligence",
        "Fintech", "Genomics", "Robotics", "Cloud Computing", "Cybersecurity"
    ]
    
    disruptive_industries = [
        "Software", "Semiconductors", "Biotechnology", "Medical Devices",
        "Internet Content & Information", "Diagnostics & Research",
        "Solar", "Electronic Components", "Information Technology Services",
        "Communication Equipment", "Computer Hardware", "Healthcare Information Services"
    ]
    
    if any(disruptive_term.lower() in sector.lower() for disruptive_term in disruptive_sectors):
        score += 2
        details.append(f"Company operates in disruptive sector: {sector}")
    elif any(disruptive_term.lower() in industry.lower() for disruptive_term in disruptive_industries):
        score += 2
        details.append(f"Company operates in innovative industry: {industry}")
    else:
        details.append(f"Company sector ({sector}) and industry ({industry}) not identified as highly disruptive")
    
    # 2. Revenue Growth Rate - Cathie Wood looks for high growth
    revenue_growth = metrics.get("revenue_growth")
    if revenue_growth is not None and not np.isnan(revenue_growth):
        if revenue_growth > 0.30:  # 30%+ annual growth
            score += 3
            details.append(f"Exceptional revenue growth rate of {revenue_growth:.1%}")
        elif revenue_growth > 0.15:  # 15-30% growth
            score += 2
            details.append(f"Strong revenue growth rate of {revenue_growth:.1%}")
        elif revenue_growth > 0.05:  # 5-15% growth
            score += 1
            details.append(f"Moderate revenue growth rate of {revenue_growth:.1%}")
        else:
            details.append(f"Limited revenue growth rate of {revenue_growth:.1%}")
    else:
        details.append("Revenue growth data not available")
    
    # 3. Gross Margin - High margins suggest pricing power and innovation
    gross_margin = metrics.get("gross_margin")
    if gross_margin is not None and not np.isnan(gross_margin):
        if gross_margin > 0.60:  # 60%+ gross margin
            score += 2
            details.append(f"Exceptional gross margin of {gross_margin:.1%}")
        elif gross_margin > 0.40:  # 40-60% gross margin
            score += 1
            details.append(f"Strong gross margin of {gross_margin:.1%}")
        else:
            details.append(f"Moderate gross margin of {gross_margin:.1%}")
    else:
        details.append("Gross margin data not available")
    
    # 4. Market Position - Check if company is a leader in its space
    market_cap = metrics.get("market_cap")
    if market_cap is not None and not np.isnan(market_cap):
        if market_cap > 100e9:  # $100B+ market cap
            details.append("Large established company, may be driving innovation")
        elif market_cap > 10e9:  # $10B-$100B
            score += 1
            details.append("Mid-sized company with potential for market disruption")
        elif market_cap > 1e9:  # $1B-$10B
            score += 2
            details.append("Growth-stage company with high disruptive potential")
        else:
            details.append("Smaller company, disruptive potential uncertain")
    else:
        details.append("Market cap data not available")
    
    return {"score": score, "details": "; ".join(details)}


def analyse_innovation_growth(metrics: dict, ticker: str, company_ticker=None):
    """
    Analyze the company's innovation metrics and growth trajectory.
    
    Args:
        metrics (dict): Financial metrics dictionary
        ticker (str): Stock ticker symbol
        company_ticker (yf.Ticker, optional): Ticker object to reuse
        
    Returns:
        dict: Analysis results with score and details
    """
    score = 0
    details = []
    
    # Reuse ticker object if provided, otherwise get a new one
    if company_ticker is None:
        try:
            company_ticker = get_ticker(ticker)
        except Exception as e:
            return {"score": 0, "details": f"Failed to get ticker data: {str(e)}"}
    
    try:
        financials = company_ticker.financials
        if financials is None or financials.empty:
            return {"score": 0, "details": "No financial data available"}
    except Exception as e:
        return {"score": 0, "details": f"Error retrieving financial data: {str(e)}"}
    
    if not metrics:
        return {"score": 0, "details": "Insufficient data for innovation growth analysis"}
    
    # 1. R&D Spending Growth and Intensity
    # Try to find R&D expenses in the financials
    rd_expense_df = safe_get_row(financials, "Research Development", 
                                ["Research And Development", "R&D Expense"])
    
    if rd_expense_df is not None:
        rd_expenses = filter_valid_values(rd_expense_df)
        
        if len(rd_expenses) >= 2:
            try:
                # Note: yfinance data is typically in reverse chronological order (newest first)
                latest_rd = rd_expenses[0]
                earliest_rd = rd_expenses[-1]
                
                if earliest_rd > 0:  # Avoid division by zero
                    rd_growth = (latest_rd - earliest_rd) / earliest_rd
                    
                    if rd_growth > 0.25:  # 25%+ R&D growth
                        score += 3
                        details.append(f"Strong R&D spending growth of {rd_growth:.1%}")
                    elif rd_growth > 0.10:  # 10-25% R&D growth
                        score += 2
                        details.append(f"Solid R&D spending growth of {rd_growth:.1%}")
                    elif rd_growth > 0:  # 0-10% R&D growth
                        score += 1
                        details.append(f"Modest R&D spending growth of {rd_growth:.1%}")
                    else:
                        details.append(f"Declining R&D spending: {rd_growth:.1%}")
                else:
                    details.append("Cannot calculate R&D growth (zero or negative base value)")
            except (IndexError, ZeroDivisionError, Exception) as e:
                details.append(f"Error calculating R&D growth: {str(e)}")
            
            # Check R&D intensity (R&D as % of revenue)
            try:
                revenue_df = safe_get_row(financials, "Total Revenue", ["Revenue", "Revenues"])
                
                if revenue_df is not None:
                    revenues = filter_valid_values(revenue_df)
                    
                    if revenues and len(revenues) >= 1 and revenues[0] > 0:
                        rd_intensity = rd_expenses[0] / revenues[0]
                        
                        if rd_intensity > 0.15:  # 15%+ of revenue on R&D
                            score += 2
                            details.append(f"High R&D intensity: {rd_intensity:.1%} of revenue")
                        elif rd_intensity > 0.08:  # 8-15% of revenue on R&D
                            score += 1
                            details.append(f"Moderate R&D intensity: {rd_intensity:.1%} of revenue")
                        else:
                            details.append(f"Lower R&D intensity: {rd_intensity:.1%} of revenue")
                    else:
                        details.append("Cannot calculate R&D intensity (revenue data invalid)")
                else:
                    details.append("Revenue data not available for R&D intensity calculation")
            except (IndexError, ZeroDivisionError, Exception) as e:
                details.append(f"Error calculating R&D intensity: {str(e)}")
        else:
            details.append("Insufficient R&D data for trend analysis (need at least 2 periods)")
    else:
        details.append("R&D expense data not available in financial statements")
    
    # 2. Gross Profit Growth - Indicator of scaling innovation
    gross_profit_df = safe_get_row(financials, "Gross Profit")
    
    if gross_profit_df is not None:
        gross_profits = filter_valid_values(gross_profit_df)
        
        if len(gross_profits) >= 2:
            try:
                # Note: yfinance data is typically in reverse chronological order (newest first)
                latest_gp = gross_profits[0]
                earliest_gp = gross_profits[-1]
                
                if earliest_gp > 0:  # Avoid division by zero
                    gp_growth = (latest_gp - earliest_gp) / earliest_gp
                    
                    if gp_growth > 0.30:  # 30%+ gross profit growth
                        score += 2
                        details.append(f"Strong gross profit growth of {gp_growth:.1%}")
                    elif gp_growth > 0.15:  # 15-30% gross profit growth
                        score += 1
                        details.append(f"Solid gross profit growth of {gp_growth:.1%}")
                    elif gp_growth > 0:  # 0-15% gross profit growth
                        details.append(f"Modest gross profit growth of {gp_growth:.1%}")
                    else:
                        details.append(f"Declining gross profit: {gp_growth:.1%}")
                else:
                    details.append("Cannot calculate gross profit growth (zero or negative base value)")
            except (IndexError, ZeroDivisionError, Exception) as e:
                details.append(f"Error calculating gross profit growth: {str(e)}")
        else:
            details.append("Insufficient gross profit data for trend analysis (need at least 2 periods)")
    else:
        details.append("Gross profit data not available in financial statements")
    
    # 3. Operating Margin Trend - Check if scaling is improving profitability
    operating_income_df = safe_get_row(financials, "Operating Income", ["Operating Profit"])
    revenue_df = safe_get_row(financials, "Total Revenue", ["Revenue", "Revenues"])
    
    if operating_income_df is not None and revenue_df is not None:
        op_incomes = filter_valid_values(operating_income_df)
        revenues = filter_valid_values(revenue_df)
        
        if len(op_incomes) >= 2 and len(revenues) >= 2 and len(op_incomes) == len(revenues):
            try:
                # Calculate operating margins for earliest and latest periods
                latest_margin = op_incomes[0] / revenues[0] if revenues[0] > 0 else None
                earliest_margin = op_incomes[-1] / revenues[-1] if revenues[-1] > 0 else None
                
                if latest_margin is not None and earliest_margin is not None:
                    margin_change = latest_margin - earliest_margin
                    
                    if margin_change > 0.05:  # 5+ percentage point improvement
                        score += 2
                        details.append(f"Improving operating margin: +{margin_change:.1%} points")
                    elif margin_change > 0:  # 0-5 percentage point improvement
                        score += 1
                        details.append(f"Slightly improving operating margin: +{margin_change:.1%} points")
                    else:
                        details.append(f"Declining operating margin: {margin_change:.1%} points")
                else:
                    details.append("Cannot calculate operating margin trend (division by zero)")
            except (IndexError, ZeroDivisionError, Exception) as e:
                details.append(f"Error calculating operating margin trend: {str(e)}")
        else:
            details.append("Insufficient data for operating margin trend analysis")
    else:
        details.append("Operating income or revenue data not available for margin analysis")
    
    return {"score": score, "details": "; ".join(details)}


def analyse_cathie_wood_valuation(metrics: dict, ticker: str, growth_rate: float = 0.20, 
                                 discount_rate: float = 0.15, terminal_multiple: float = 25, 
                                 projection_years: int = 5, company_ticker=None):
    """
    Analyze the company's valuation using Cathie Wood's growth-focused approach.
    
    Args:
        metrics (dict): Financial metrics dictionary
        ticker (str): Stock ticker symbol
        growth_rate (float): Annual growth rate for projections (default 20%)
        discount_rate (float): Discount rate for present value calculations (default 15%)
        terminal_multiple (float): Multiple for terminal value calculation (default 25x)
        projection_years (int): Number of years to project cash flows (default 5)
        company_ticker (yf.Ticker, optional): Ticker object to reuse
        
    Returns:
        dict: Analysis results with score and details
    """
    # Reuse ticker object if provided, otherwise get a new one
    if company_ticker is None:
        try:
            company_ticker = get_ticker(ticker)
        except Exception as e:
            return {"score": 0, "details": f"Failed to get ticker data: {str(e)}"}
    
    if not metrics:
        return {"score": 0, "details": "Insufficient data for valuation analysis"}
    
    score = 0
    details = []
    
    # Get key metrics for valuation
    revenue = metrics.get("revenue")
    market_cap = metrics.get("market_cap")
    
    if revenue is None or np.isnan(revenue) or revenue <= 0:
        return {"score": 0, "details": "Revenue data not available or invalid"}
    
    if market_cap is None or np.isnan(market_cap) or market_cap <= 0:
        return {"score": 0, "details": "Market cap data not available or invalid"}
    
    # 1. Price-to-Sales Ratio - Cathie Wood often looks at this for growth companies
    ps_ratio = market_cap / revenue
    
    if ps_ratio < 5:
        score += 2
        details.append(f"Attractive P/S ratio of {ps_ratio:.2f}x")
    elif ps_ratio < 10:
        score += 1
        details.append(f"Moderate P/S ratio of {ps_ratio:.2f}x")
    elif ps_ratio < 20:
        details.append(f"High but potentially justified P/S ratio of {ps_ratio:.2f}x")
    else:
        details.append(f"Very high P/S ratio of {ps_ratio:.2f}x")
    
    # 2. Revenue Growth-Adjusted Valuation
    # Cathie Wood often considers growth rates in context of valuation
    revenue_growth = metrics.get("revenue_growth")
    
    if revenue_growth is not None and not np.isnan(revenue_growth) and revenue_growth > 0:
        growth_adjusted_ps = ps_ratio / revenue_growth
        
        if growth_adjusted_ps < 1:
            score += 2
            details.append(f"Excellent growth-adjusted P/S of {growth_adjusted_ps:.2f}")
        elif growth_adjusted_ps < 2:
            score += 1
            details.append(f"Good growth-adjusted P/S of {growth_adjusted_ps:.2f}")
        else:
            details.append(f"Higher growth-adjusted P/S of {growth_adjusted_ps:.2f}")
    else:
        details.append("Cannot calculate growth-adjusted valuation (growth data unavailable)")
    
    # 3. Future Value Projection - Cathie Wood's approach to valuation
    try:
        # Project future revenue
        future_revenue = revenue * (1 + growth_rate) ** projection_years
        
        # Apply a future multiple to the projected revenue
        future_market_cap = future_revenue * terminal_multiple
        
        # Discount back to present value
        present_value = future_market_cap / ((1 + discount_rate) ** projection_years)
        
        # Calculate implied upside/downside
        potential_return = (present_value - market_cap) / market_cap
        
        details.append(f"Current revenue: ${revenue:,.0f}")
        details.append(f"Projected {projection_years}-year revenue: ${future_revenue:,.0f}")
        details.append(f"Implied future market cap: ${future_market_cap:,.0f}")
        details.append(f"Present value: ${present_value:,.0f}")
        
        if potential_return > 1.0:  # 100%+ upside
            score += 3
            details.append(f"Significant upside potential: {potential_return:.1%}")
        elif potential_return > 0.5:  # 50-100% upside
            score += 2
            details.append(f"Strong upside potential: {potential_return:.1%}")
        elif potential_return > 0.2:  # 20-50% upside
            score += 1
            details.append(f"Moderate upside potential: {potential_return:.1%}")
        elif potential_return > 0:  # 0-20% upside
            details.append(f"Limited upside potential: {potential_return:.1%}")
        else:
            details.append(f"Potential downside: {potential_return:.1%}")
    except Exception as e:
        details.append(f"Error in future value projection: {str(e)}")
    
    return {"score": score, "details": "; ".join(details)}


if __name__ == "__main__":
    analysis = calculate_cathie_wood_analysis_data("TSLA")
    print(analysis)    
