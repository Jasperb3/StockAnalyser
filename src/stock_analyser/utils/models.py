from typing import List, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field

#######################
# Base Models
#######################

class ReportState(BaseModel):
    """
    Represents the overall state of the stock analysis report.
    """
    company_name: str = Field(default="", description="Name of the company")
    """Example: "Apple Inc." """
    
    company_ticker: str = Field(default="", description="Ticker symbol of the company")
    """Example: "AAPL" """
    
    company_website: str = Field(default="", description="Website of the company")
    """Example: "https://www.apple.com/" """
    
    industry: str = Field(default="", description="Industry the company belongs to")
    """Example: "Technology" """
    
    executive_summary: str = Field(default="", description="Executive summary of the report")
    """Example: "This report provides a comprehensive analysis of Apple Inc.'s financial performance, market position, and future prospects." """
    
    market_and_industry_section: str = Field(default="", description="Market and industry context section")
    """Example: "The technology sector continues to show strong growth, with particular momentum in AI and cloud services." """

    financial_data_section: str = Field(default="", description="Financial data section")
    """Example: "Revenue has grown by 15% YoY, with operating margins expanding to 30%." """
    
    competitor_landscape_section: str = Field(default="", description="Competitor landscape section")
    """Example: "The technology sector continues to show strong growth, with particular momentum in AI and cloud services." """
    
    analyst_insights_section: str = Field(default="", description="Analyst insights section")
    """Example: "Analysts maintain a consensus Buy rating with a mean price target of $200." """

    benjamin_graham_section: str = Field(default="", description="Benjamin Graham section")
    """Example: "We recommend a Buy rating with a 12-month price target of $210." """

    warren_buffet_section: str = Field(default="", description="Warren Buffet section")
    """Example: "We recommend a Buy rating with a 12-month price target of $210." """

    cathie_wood_section: str = Field(default="", description="Cathie Wood section")
    """Example: "We recommend a Buy rating with a 12-month price target of $210." """

    charlie_munger_section: str = Field(default="", description="Charlie Munger section")
    """Example: "We recommend a Buy rating with a 12-month price target of $210." """

    bill_ackman_section: str = Field(default="", description="Bill Ackman section")
    """Example: "We recommend a Buy rating with a 12-month price target of $210." """
    
    risk_analysis_section: str = Field(default="", description="Risk analysis section")
    """Example: "Key risks include supply chain disruptions and increasing regulatory scrutiny." """
    
    investment_recommendations_section: str = Field(default="", description="Investment recommendations section")
    """Example: "We recommend a Buy rating with a 12-month price target of $210." """

    news_and_research_section: str = Field(default="", description="News and research section")
    """Example: "The company is expected to benefit from the growing demand in wearables and services segments." """
    
    future_outlook_section: str = Field(default="", description="Future outlook section")
    """Example: "The company is expected to benefit from the growing demand in wearables and services segments." """

    conclusion: str = Field(default="", description="Conclusion section")
    """Example: "In conclusion, we recommend a Buy rating with a 12-month price target of $210." """
    
    date_iso: str = datetime.now().strftime("%Y-%m-%d")
    """Example: "2024-03-20" """

    date_us: str = datetime.now().strftime("%A, %B %d, %Y")
    """Example: "Wednesday, March 20, 2024" """

    total_token_usage: int = Field(default=0, description="Total token usage for the report")
    """Example: 1000 """


class ReportCritique(BaseModel):
    """
    Represents a critique of a report section or content.
    """
    content: str
    """Example: "The financial analysis lacks detailed cash flow projections." """
    
    critique: str
    """Example: "The analysis should include a detailed breakdown of operating cash flows and their growth trends over the past 5 years." """


#######################
# Market & Industry Models
#######################

class MarketIndustryContextModel(BaseModel):
    """
    Defines the scope and structure for the Market & Industry Context section.
    """
    company_information: str
    """Example: "Apple Inc. is a technology company that manufactures and sells smartphones, tablets, and computers." """

    management_information: str
    """Example: "Apple Inc. is managed by Tim Cook, who has been the CEO since 2011." """
    
    market_overview: Optional[str] = None
    """Example: "The global smartphone market reached $400B in 2023, growing at 8% CAGR." """
    
    industry_trends: Optional[str] = None
    """Example: "5G adoption is accelerating with 65% of new devices supporting the technology." """
    
    industry_growth_forecasts: Optional[str] = None
    """Example: "The sector is projected to grow at 12% CAGR through 2028, driven by AI and IoT integration." """

    industry_leaders: Optional[str] = None
    """Example: "Apple leads with 28% market share, followed by Samsung (22%) and Xiaomi (12%)." """
    
    macroeconomic_factors: Optional[str] = None
    """Example: "Rising interest rates and inflation may impact consumer discretionary spending in H2 2024." """
    
    regulatory_environment: Optional[str] = None
    """Example: "New EU digital market regulations expected to impact app store revenue models by Q4 2024." """


class Competitor(BaseModel):
    """
    Represents a competitor company.
    """
    company_name: str
    """Example: "Samsung Electronics Co., Ltd." """
    
    ticker: str
    """Example: "SSNLF" """


class Ticker(BaseModel):
    """
    Represents a ticker symbol.
    """
    ticker: str
    """Example: "SSNLF" """


class Article(BaseModel):
    """
    Represents a news article.
    """
    title: str
    url: str
    article_text: str


class CompetitorList(BaseModel):
    """
    Represents a list of competitor companies.
    """
    competitors: List[Competitor]
    """
    Example: [
        {"company_name": "Samsung Electronics Co., Ltd.", "ticker": "SSNLF"},
        {"company_name": "Xiaomi Corporation", "ticker": "1810.HK"},
        {"company_name": "Huawei Technologies Co., Ltd.", "ticker": "Private"}
    ]
    """


class CompetitorFinancialData(BaseModel):
    """
    Represents the financial data of a competitor company.
    """
    competitors: List[Competitor]
    """Example: [
        {"company_name": "Samsung Electronics Co., Ltd.", "ticker": "SSNLF"},
        {"company_name": "Xiaomi Corporation", "ticker": "1810.HK"},
        {"company_name": "Huawei Technologies Co., Ltd.", "ticker": "Private"}
    ]"""

    revenue_growth_yoy: Optional[float] = None
    """Example: 0.10 (representing 10% growth)"""

    net_margin: Optional[float] = None
    """Example: 0.20 (representing a 20% net margin)"""

    roe: Optional[float] = None
    """Example: 0.25 (representing a 25% return on equity)"""

    pe_ratio: Optional[float] = None
    """Example: 25.5"""

    ps_ratio: Optional[float] = None
    """Example: 5.2"""

    news_articles: Optional[List[Article]] = None
    """Example: [
        {"title": "Samsung Electronics Co., Ltd. announces new product launch", "url": "https://www.samsung.com/us/news/press-releases/2024/03/samsung-electronics-co-ltd-announces-new-product-launch", "article_text": "Samsung Electronics Co., Ltd. announces new product launch"},
        {"title": "Xiaomi Corporation reports strong quarterly earnings", "url": "https://www.xiaomi.com/news/press-releases/2024/03/xiaomi-corporation-reports-strong-quarterly-earnings", "article_text": "Xiaomi Corporation reports strong quarterly earnings"},
        {"title": "Huawei Technologies Co., Ltd. faces regulatory scrutiny", "url": "https://www.huawei.com/news/press-releases/2024/03/huawei-technologies-co-ltd-faces-regulatory-scrutiny", "article_text": "Huawei Technologies Co., Ltd. faces regulatory scrutiny"}
    ]"""


class CompetitorSection(BaseModel):
    """
    Represents the competitor section of a stock analysis report.
    """
    competitors: List[Competitor]
    """Example: [
        {"company_name": "Samsung Electronics Co., Ltd.", "ticker": "SSNLF"},
        {"company_name": "Xiaomi Corporation", "ticker": "1810.HK"},
        {"company_name": "Huawei Technologies Co., Ltd.", "ticker": "Private"}
    ]"""

    content: str
    """Example: "The company is expected to benefit from the growing demand in wearables and services segments." """


#######################
# Analysts Insights Models
#######################

class AnalystsInsightsModel(BaseModel):
    """
    Defines the scope and structure for the Analysts' Insights section.
    """
    analyst_recommendations: Optional[str] = None
    """Example: "Period: 2023-09-30, Strong Buy: 15, Buy: 20, Hold: 3, Sell: 1, Strong Sell: 0" """

    analyst_price_targets: Optional[str] = None
    """Example: "Current: 175.0, Target High: 250.0, Target Low: 120.0, Target Mean: 195.5, Target Median: 198.0" """

    upgrades_downgrades: Optional[str] = None
    """Example: "Date: 2024-03-15, Firm: Morgan Stanley, From Grade: Hold, To Grade: Buy" """

    institutional_and_mutual_fund_holders: Optional[str] = None
    """Example: "Date: 2024-03-01, Institution: Vanguard, Percent Held: 8.5%, Shares: 1,200,000, Value: $210,000,000, Percent Change: 2.5%" """

    insider_trading: Optional[str] = None
    """Example: "Date: 2024-03-01, Insider: John Doe, Position: CEO, Shares: 100,000, Value: $1,500,000, Text: 'Purchased shares for personal use'" """

    other_insights: Optional[str] = None
    """Example: "The company is expected to benefit from the growing demand in wearables and services segments." """


class ExpertAnalystSignal(BaseModel):
    """
    Represents an expert analyst signal.
    """
    signal: Literal["bullish", "bearish", "neutral"]
    """Example: "bullish" """
    
    confidence: float
    """Example: 0.95 """

    reasoning: str
    """Example: "The company is expected to benefit from the growing demand in wearables and services segments." """



class InvestmentRecommendationsModel(BaseModel):
    """
    Defines the scope and structure for the Investment Recommendations section.
    
    Example:
    ```json
    {
        "valuation_factors": "Trading at 25x forward P/E...",
        "fundamental_assessment": "Strong brand equity and ecosystem...",
        "market_outlook": "Beneficiary of AI and 5G trends...",
        "potential_catalysts": "New product launches in Q3...",
        "technical_considerations": "Support at $150, resistance at $180...",
        "analyst_recommendations": "Warren Buffett:Strong Buy with a target price of $210..., Cathie Wood:Strong Buy with a target price of $210...",
        "other_recommendations": "Recommend staged buying approach..."
    }
    ```
    """
    valuation_factors: Optional[str] = None
    """Example: "Trading at 25x forward P/E vs. peer average of 22x, justified by superior growth profile and margins." """
    
    fundamental_assessment: Optional[str] = None
    """Example: "Strong brand equity and ecosystem lock-in provide sustainable competitive advantages." """
    
    market_outlook: Optional[str] = None
    """Example: "Beneficiary of AI and 5G trends, with potential upside from new product categories." """
    
    potential_catalysts: Optional[str] = None
    """Example: "New product launches in Q3 2024, services bundle expansion, and potential M&A activity." """
    
    technical_considerations: Optional[str] = None
    """Example: "Support at $150, resistance at $180. Moving averages suggest positive momentum." """

    analyst_recommendations: Optional[str] = None
    """Example: "Warren Buffett:Strong Buy with a target price of $210..., Cathie Wood:Strong Buy with a target price of $210..." """
    
    other_recommendations: Optional[str] = None
    """Example: "Recommend staged buying approach: 50% position now, 25% at $160, 25% at $150." """


class ResearchQuestion(BaseModel):
    """
    Represents a research question and answer.
    """
    research_question: str
    """Example: "What is the company's market outlook?" """


class ResearchAnswer(BaseModel):
    """
    Represents a research answer.
    """
    research_answer: str
    """Example: "The company is expected to benefit from the growing demand in wearables and services segments." """


class ResearchQuestions(BaseModel):
    """
    Represents a list of research questions.
    """
    research_questions: List[ResearchQuestion]
    """Example: [
        {"research_question": "What is the company's market outlook?"},
        {"research_question": "How does the companies valuation compare to its peers?"}
    ]"""


class ResearchAnswers(BaseModel):
    """
    Represents a list of research answers.
    """
    research_answers: List[ResearchAnswer]
    """Example: [   
        {"research_answer": "The company is expected to benefit from the growing demand in wearables and services segments."},
        {"research_answer": "The company is trading at a premium compared to its peers, but this is justified by its superior growth prospects and profitability."}
    ]"""


class FutureOutlookModel(BaseModel):
    """
    Defines the scope and structure for the Future Outlook section.
    
    Example:
    ```json
    {
        "expected_industry_disruptions": "AI integration changing competitive landscape...",
        "long_term_strategic_initiatives": "5-year roadmap includes healthcare expansion...",
        "macro_trends_impact": "Demographic shifts and digital transformation...",
        "catalysts_and_triggers": "New product launches, M&A activity, and regulatory changes...",
        "other_outlook_possibilities": "Potential expansion into AR/VR..."
    }
    ```
    """
    expected_industry_disruptions: Optional[str] = None
    """Example: "AI integration changing competitive landscape; quantum computing developments on horizon." """
    
    long_term_strategic_initiatives: Optional[str] = None
    """Example: "5-year roadmap includes healthcare expansion, AR/VR ecosystem development, and autonomous systems." """
    
    macro_trends_impact: Optional[str] = None
    """Example: "Demographic shifts and digital transformation driving demand for connected devices and services." """

    catalysts_and_triggers: Optional[str] = None
    """Example: "New product launches, M&A activity, and regulatory changes." """
    
    other_outlook_possibilities: Optional[str] = None
    """Example: "Potential expansion into AR/VR hardware, autonomous vehicles, and healthcare devices." """


#######################
# Financial Analysis Models
#######################

class IncomeStatementAnalysis(BaseModel):
    """
    Represents the income statement analysis of a company, focusing on key metrics.
    """
    total_revenue: str
    """Example: "Total Revenue: $394.33B" """
    gross_profit: str
    """Example: "Gross Profit: $170.78B" """
    operating_income: str
    """Example: "Operating Income: $119.44B" """
    net_income: str
    """Example: "Net Income: $97.15B" """
    ebitda: str
    """Example: "EBITDA: $125.5B" """
    diluted_eps: str
    """Example: "Diluted EPS: $6.14" """
    revenue_growth: str
    """Example: "Revenue Growth: 8.1%" """
    earnings_growth: str
    """Example: "Earnings Growth: 9.2%" """
    gross_margins: str
    """Example: "Gross Margins: 43.3%" """
    operating_margin: str
    """Example: "Operating Margin: 30.3%" """
    profit_margin: str
    """Example: "Profit Margin: 24.6%" """

class BalanceSheetAnalysis(BaseModel):
    """
    Represents the balance sheet analysis, focusing on key liquidity and solvency metrics.
    """
    total_assets: str
    """Example: "Total Assets: $400B" """
    total_liabilities: str
    """Example: "Total Liabilities: $300B" """
    total_equity: str
    """Example: "Total Equity: $100B" """
    total_debt: str
    """Example: "Total Debt: $120.07B" """
    net_debt: str
    """Example: "Net Debt: $57.57B" """
    cash_and_equivalents: str
    """Example: "Cash and Cash Equivalents: $62.5B" """
    working_capital: str
    """Example: "Working Capital: $10B" """
    current_ratio: str
    """Example: "Current Ratio: 1.4" """
    quick_ratio: str
    """Example: "Quick Ratio: 1.2" """
    debt_to_equity: str
    """Example: "Debt-to-Equity: 1.68" """

class CashFlowAnalysis(BaseModel):
    """
    Represents the cash flow analysis, focusing on key cash flow metrics.
    """
    operating_cash_flow: str
    """Example: "Operating Cash Flow: $114.5B" """
    free_cash_flow: str
    """Example: "Free Cash Flow: $90.15B" """
    capital_expenditure: str
    """Example: "Capital Expenditure: $24.35B" """
    cash_dividends_paid: str
    """Example: "Cash Dividends Paid: $14.85B" """

class ValuationMetrics(BaseModel):
    """
    Represents key valuation metrics for a company.
    """
    market_cap: str
    """Example: "Market Cap: $2.75T" """
    enterprise_value: str
    """Example: "Enterprise Value: $2.65T" """
    pe_ratio: str
    """Example: "P/E Ratio (Trailing): 28.5" """
    forward_pe_ratio: str
    """Example: "Forward P/E Ratio: 25.3" """
    price_to_sales: str
    """Example: "Price to Sales (TTM): 7.2" """
    price_to_book: str
    """Example: "Price to Book: 35.7" """
    enterprise_to_ebitda: str
    """Example: "Enterprise Value to EBITDA: 20.3" """
    enterprise_to_revenue: str
    """Example: "Enterprise Value to Revenue: 6.7" """

class PriceMetrics(BaseModel):
    """
    Represents the price metrics of a company.
    """
    current_price: str
    """Example: "Current Price: $175.50" """
    fifty_two_week_high: str
    """Example: "52-Week High: $198.23" """
    fifty_two_week_low: str
    """Example: "52-Week Low: $124.17" """
    fifty_day_average: str
    """Example: "50-Day Average: $182.35" """
    two_hundred_day_average: str
    """Example: "200-Day Average: $170.22" """

class ReturnMetrics(BaseModel):
    """
    Represents key return metrics.
    """
    return_on_equity: str
    """Example: "Return on Equity: 160.09%" """
    return_on_assets: str
    """Example: "Return on Assets: 28.30%" """
    return_on_invested_capital: str
    """Example: "Return on Invested Capital: 33.5%" """

class DividendMetrics(BaseModel):
    """
    Represents key dividend metrics.
    """
    dividend_yield: str
    """Example: "Dividend Yield: 0.55%" """
    dividend_rate: str
    """Example: "Annual Dividend Rate: $0.96" """
    payout_ratio: str
    """Example: "Payout Ratio: 15.20%" """

class AnalystTargets(BaseModel):
    """
    Represents analyst targets.
    """
    analyst_target_price: str
    """Example: "Mean Target: $210.50" """
    average_analyst_rating: str
    """Example: "Average Analyst Rating: Buy" """

class OwnershipStructure(BaseModel):
    """
    Represents key ownership structure data.
    """
    shares_outstanding: str
    """Example: "Shares Outstanding: 15.7B" """
    held_percent_insiders: str
    """Example: "Held Percent Insiders: 0.08%" """
    held_percent_institutions: str
    """Example: "Held Percent Institutions: 60.12%" """

class FinancialData(BaseModel):
    """
    Represents the complete financial data analysis of a company.
    """
    income_statement_analysis: IncomeStatementAnalysis
    balance_sheet_analysis: BalanceSheetAnalysis
    cash_flow_analysis: CashFlowAnalysis
    valuation_metrics: ValuationMetrics
    price_metrics: PriceMetrics
    return_metrics: ReturnMetrics
    dividend_metrics: DividendMetrics
    analyst_targets: AnalystTargets
    ownership_structure: OwnershipStructure


#######################
# Risk Analysis Models
#######################

class RiskAnalysisModel(BaseModel):
    """
    Defines the scope and structure for the Risk Analysis section of a stock analysis report.
    
    Example:
    ```json
    {
        "market_volatility": "High interest rate sensitivity with beta of 1.2...",
        "regulatory_hurdles": "Antitrust investigations in EU and US...",
        "esg_risks": "Carbon footprint and supply chain sustainability concerns...",
        "company_specific_issues": "Supply chain concentration in Asia...",
        "other_risks": "Cybersecurity threats and talent retention challenges..."
    }
    ```
    """
    market_volatility: Optional[str] = None
    """Example: "High interest rate sensitivity with beta of 1.2, exposed to tech sector rotation risk." """
    
    regulatory_hurdles: Optional[str] = None
    """Example: "Antitrust investigations in EU and US markets could impact app store revenue model." """

    esg_risks: Optional[str] = None
    """Example: "Carbon footprint and supply chain sustainability concerns." """
    
    company_specific_issues: Optional[str] = None
    """Example: "Supply chain concentration in Asia poses geopolitical risk; 70% of manufacturing in single region." """
    
    other_risks: Optional[str] = None
    """Example: "Cybersecurity threats and talent retention challenges in competitive labor market." """


class Risk(BaseModel):
    """
    Represents a risk.
    """
    risk: str
    """Example: "Market volatility" """


class RiskList(BaseModel):
    """
    Represents a list of risks.
    """
    risk_list: List[Risk]


class RiskSeverity(BaseModel):
    """
    Represents the severity of a risk.
    """
    risk: str
    """Example: "Market volatility" """
    
    severity: int
    """Example: 6 """


class RiskSeverityList(BaseModel):
    """
    Represents a list of risks.
    """
    risk_severity_list: List[RiskSeverity]


#######################
# News & Research Models
####################### 

class Article(BaseModel):
    """
    Represents a news article.
    """
    title: str
    url: str
    article_text: str


class NewsAndResearchModel(BaseModel):
    """
    Represents the news and research analysis of a company.
    """
    articles: List[Article]




