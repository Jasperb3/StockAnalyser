from typing import List, Optional, Literal
from datetime import datetime
from pathlib import Path
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

    trends_section: str = Field(default="", description="Trends section")
    """Example: "The technology sector continues to show strong growth, with particular momentum in AI and cloud services." """
    
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

    stanley_druckenmiller_section: str = Field(default="", description="Stanley Druckenmiller section")
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

    report_path_md: Path = Field(default="", description="Path to the markdown report")
    """Example: "reports/2024-03-20_stock_analysis_report.md" """

    report_path_pdf: Path = Field(default="", description="Path to the pdf report")
    """Example: "reports/2024-03-20_stock_analysis_report.pdf" """

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

    key_products_services: str
    """Example: "The company's key products include the iPhone, iPad, and MacBook." """

    business_model: str
    """Example: "Apple Inc. is a technology company that manufactures and sells smartphones, tablets, and computers." """
    
    market_overview: Optional[str] = None
    """Example: "The global smartphone market reached $400B in 2023, growing at 8% CAGR due to high demand for new technology." """
    
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


class CompetitorData(BaseModel):
    """
    Represents the data of a competitor company.
    """
    competitor_name: str
    """Example: "Samsung Electronics Co., Ltd." """

    competitor_ticker: str
    """Example: "SSNLF" """

    revenue_growth_yoy: float = None
    """Example: 0.10 (representing 10% growth)"""

    net_margin: float = None
    """Example: 0.20 (representing a 20% net margin)"""

    return_on_equity: float = None
    """Example: 0.25 (representing a 25% return on equity)"""

    price_to_earnings_ratio: float = None
    """Example: 25.5"""

    price_to_sales_ratio: float = None
    """Example: 5.2"""

    price_to_book_ratio: float = None
    """Example: 1.5"""

    recent_news_articles: List[Article] = None
    """Example: [
        {"title": "Samsung Electronics Co., Ltd. announces new product launch", "url": "https://www.samsung.com/us/news/press-releases/2024/03/samsung-electronics-co-ltd-announces-new-product-launch", "article_text": "Samsung Electronics Co., Ltd. announces new product launch"},
        {"title": "Xiaomi Corporation reports strong quarterly earnings", "url": "https://www.xiaomi.com/news/press-releases/2024/03/xiaomi-corporation-reports-strong-quarterly-earnings", "article_text": "Xiaomi Corporation reports strong quarterly earnings"},
        {"title": "Huawei Technologies Co., Ltd. faces regulatory scrutiny", "url": "https://www.huawei.com/news/press-releases/2024/03/huawei-technologies-co-ltd-faces-regulatory-scrutiny", "article_text": "Huawei Technologies Co., Ltd. faces regulatory scrutiny"}
    ]"""
    
    
class CompetitorFinancialData(BaseModel):
    """
    Represents the financial data of a competitor company.
    """
    competitors: List[CompetitorData]
    """Example: [
        {"competitor_name": "Samsung Electronics Co., Ltd.", "competitor_ticker": "SSNLF", "revenue_growth_yoy": 0.10, "net_margin": 0.20, "roe": 0.25, "pe_ratio": 25.5, "ps_ratio": 5.2, "recent_news_articles": [
            {"title": "Samsung Electronics Co., Ltd. announces new product launch", "url": "https://www.samsung.com/us/news/press-releases/2024/03/samsung-electronics-co-ltd-announces-new-product-launch", "article_text": "Samsung Electronics Co., Ltd. announces new product launch"},
            {"title": "Xiaomi Corporation reports strong quarterly earnings", "url": "https://www.xiaomi.com/news/press-releases/2024/03/xiaomi-corporation-reports-strong-quarterly-earnings", "article_text": "Xiaomi Corporation reports strong quarterly earnings"},
            {"title": "Huawei Technologies Co., Ltd. faces regulatory scrutiny", "url": "https://www.huawei.com/news/press-releases/2024/03/huawei-technologies-co-ltd-faces-regulatory-scrutiny", "article_text": "Huawei Technologies Co., Ltd. faces regulatory scrutiny"}
        ]}
    ]"""
    
    

# class CompetitorFinancialData(BaseModel):
#     """
#     Represents the financial data of a competitor company.
#     """
#     competitors: List[Competitor]
#     """Example: [
#         {"company_name": "Samsung Electronics Co., Ltd.", "ticker": "SSNLF"},
#         {"company_name": "Xiaomi Corporation", "ticker": "1810.HK"},
#         {"company_name": "Huawei Technologies Co., Ltd.", "ticker": "Private"}
#     ]"""

#     revenue_growth_yoy: Optional[float] = None
#     """Example: 0.10 (representing 10% growth)"""

#     net_margin: Optional[float] = None
#     """Example: 0.20 (representing a 20% net margin)"""

#     roe: Optional[float] = None
#     """Example: 0.25 (representing a 25% return on equity)"""

#     pe_ratio: Optional[float] = None
#     """Example: 25.5"""

#     ps_ratio: Optional[float] = None
#     """Example: 5.2"""

#     news_articles: Optional[List[Article]] = None
#     """Example: [
#         {"title": "Samsung Electronics Co., Ltd. announces new product launch", "url": "https://www.samsung.com/us/news/press-releases/2024/03/samsung-electronics-co-ltd-announces-new-product-launch", "article_text": "Samsung Electronics Co., Ltd. announces new product launch"},
#         {"title": "Xiaomi Corporation reports strong quarterly earnings", "url": "https://www.xiaomi.com/news/press-releases/2024/03/xiaomi-corporation-reports-strong-quarterly-earnings", "article_text": "Xiaomi Corporation reports strong quarterly earnings"},
#         {"title": "Huawei Technologies Co., Ltd. faces regulatory scrutiny", "url": "https://www.huawei.com/news/press-releases/2024/03/huawei-technologies-co-ltd-faces-regulatory-scrutiny", "article_text": "Huawei Technologies Co., Ltd. faces regulatory scrutiny"}
#     ]"""


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

class ImpliedSharePrice(BaseModel):
    """
    Represents the implied share price of a stock.
    """
    implied_share_price: float
    """Example: 210.50"""

    assumptions: List[str]
    """Example: [
        "Number of years of projections = {n}",
        "Outstanding Shares = {OutShares}",
        "Revenue Growth (Terminal) = {revenue_growth_T:.2%}",
        "EBIT (Earnings Before Interest and Taxes) / Sales = {ebit_perc_T:.2%}",
        "Tax Percentage of EBIT = {tax_perc_T:.2%}",
        "Depreciation and Amortization / Sales = {dna_perc_T:.2%}",
        "Capital Expenditures / Sales = {capex_perc_T:.2%}",
        "Change in Net Working Capital / Sales = {nwc_perc_T:.2%}",
        "Weighted Average Cost of Capital = {WACC:.2%}",
        "Terminal Growth Rate = {TGR:.2%}"
    ]
    """

class AnalystsInsightsModel(BaseModel):
    """
    Defines the scope and structure for the Analysts' Insights section.
    """
    analyst_recommendations: Optional[str] = None
    """Example: "Period: 2023-09-30, Strong Buy: 15, Buy: 20, Hold: 3, Sell: 1, Strong Sell: 0" """

    analyst_price_targets: Optional[str] = None
    """Example: "Current: 175.0, Target High: 250.0, Target Low: 120.0, Target Mean: 195.5, Target Median: 198.0" """

    average_analyst_rating: str
    """Example: "Average Analyst Rating: Buy" """

    estimated_intrinsic_value_per_share: Optional[ImpliedSharePrice] = None
    """Example: 
    ```json
    {
        "estimated_intrinsic_value_per_share": 210.50,
        "assumptions": [
            "Number of years of projections = 10",
            "Outstanding Shares = 1000000000",
            "Revenue Growth (Terminal) = 0.03",
            "EBIT (Earnings Before Interest and Taxes) / Sales = 0.25",
            "Tax Percentage of EBIT = 0.21",
            "Depreciation and Amortization / Sales = 0.05",
            "Capital Expenditures / Sales = 0.06",
            "Change in Net Working Capital / Sales = 0.01",
            "Weighted Average Cost of Capital = 0.08",
            "Terminal Growth Rate = 0.03"
        ]
    }
    ```
    """

    growth_estimates: Optional[str] = None
    """Example: "Current Year: 10.0%, Next Quarter: 15.0%, Next 5 Years (per annum): 12.5%" """

    revenue_estimates: Optional[str] = None
    """Example: "Current Year: 10.0%, Next Quarter: 15.0%, Next 5 Years (per annum): 12.5%" """

    earnings_estimates: Optional[str] = None
    """Example: "Current Year: 10.0%, Next Quarter: 15.0%, Next 5 Years (per annum): 12.5%" """

    upgrades_downgrades: Optional[str] = None
    """Example: "Date: 2024-03-15, Firm: Morgan Stanley, From Grade: Hold, To Grade: Buy" """

    shares_outstanding: str
    """Example: "Shares Outstanding: 15.7B" """

    held_percent_insiders: str
    """Example: "Held Percent Insiders: 0.08%" """

    held_percent_institutions: str
    """Example: "Held Percent Institutions: 60.12%" """

    institutional_and_mutual_fund_holders: Optional[str] = None
    """Example: "Date: 2024-03-01, Institution: Vanguard, Percent Held: 8.5%, Shares: 1,200,000, Value: $210,000,000, Percent Change: 2.5%" """
    
    insider_holders: Optional[List[str]] = None
    """Example: [
        "Insider: ADAMS KATHERINE L, Position: General Counsel, Shares owned directly: 179,043",
        "Insider: TONYIFY ADAM B, Position: General Secretary, Shares owned directly: 343,678"
    ]"""

    insider_purchases_last_6m: Optional[str] = None
    """Example: "Shares: 1916902.0, Transactions: 3" """

    insider_sales_last_6m: Optional[str] = None
    """Example: "Shares: 3301125.0, Transactions: 24" """

    net_insider_shares_purchased_sold_last_6m: Optional[str] = None
    """Example: "Shares: -1384223.0, Transactions: 27" """

    total_insider_shares_held: Optional[str] = None
    """Example: "Shares: 7508458.0" """

    percent_net_shares_purchased_sold_last_6m: Optional[str] = None
    """Example: "-0.156" """

    percent_buy_shares: Optional[str] = None
    """Example: "0.216" """

    percent_sell_shares: Optional[str] = None
    """Example: "0.371" """

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
        "strengths": "Strong brand equity and ecosystem lock-in, High customer retention and loyalty...",
        "weaknesses": "Supply chain concentration in Asia, Regulatory scrutiny and competition from larger players...",
        "opportunities": "Expansion into new markets and product categories, Potential for AR/VR hardware development...",
        "threats": "Regulatory scrutiny and competition from larger players, Supply chain concentration in Asia...",
        "other_outlook_possibilities": "Potential expansion into AR/VR hardware, autonomous vehicles, and healthcare devices..."
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

    strengths: Optional[List[str]] = None
    """Example: ["Strong brand equity and ecosystem lock-in.", "High customer retention and loyalty."] """

    weaknesses: Optional[List[str]] = None
    """Example: ["Supply chain concentration in Asia.", "Regulatory scrutiny and competition from larger players."] """

    opportunities: Optional[List[str]] = None
    """Example: ["Expansion into new markets and product categories.", "Potential for AR/VR hardware development."] """

    threats: Optional[List[str]] = None
    """Example: ["Regulatory scrutiny and competition from larger players.", "Supply chain concentration in Asia."] """
    
    other_outlook_possibilities: Optional[List[str]] = None
    """Example: ["Potential expansion into AR/VR hardware, autonomous vehicles, and healthcare devices.", "Potential for AR/VR hardware development."] """


#######################
# Financial Analysis Models
#######################

class IncomeStatementAnalysis(BaseModel):
    """
    Represents the income statement analysis of a company, focusing on key metrics.
    """
    net_income: str
    """Example: "Net Income: $97.15B" """
    total_revenue: str
    """Example: "Total Revenue: $394.33B" """
    gross_profit: str
    """Example: "Gross Profit: $170.78B" """
    operating_income: str
    """Example: "Operating Income: $119.44B" """
    ebitda: str
    """Example: "EBITDA: $125.5B" """
    basic_eps: str
    """Example: "Basic EPS: $6.25" """
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
    total_cash: str
    """Example: "Total Cash: $100.00B" """
    total_debt: str
    """Example: "Total Debt: $120.07B" """
    free_cash_flow: str
    """Example: "Free Cash Flow: $90.15B" """
    operating_cash_flow: str
    """Example: "Operating Cash Flow: $114.5B" """
    capital_expenditure: str
    """Example: "Capital Expenditure: $24.35B" """
    repayment_of_debt: str
    """Example: "Repayment of Debt: $10.5B" """
    debt_issuance: str
    """Example: "Debt Issuance: None" """
    cash_changes: str
    """Example: "Changes in Cash: $15.25B" """
    investing_cash_flow: str
    """Example: "Investing Cash Flow: $-45.5B" """
    financing_cash_flow: str
    """Example: "Financing Cash Flow: $30.25B" """
    repurchases_of_stock: str
    """Example: "Repurchases of Stock: $25.75B" """

class GrowthMetrics(BaseModel):
    """
    Represents key growth metrics.
    """
    revenue_growth: str
    """Example: "Revenue Growth: 8.1%" """
    earnings_growth: str
    """Example: "Earnings Growth: 9.2%" """
    five_year_revenue_growth_rate: str
    """Example: "5-Year Revenue Growth Rate: 12.5%" """
    two_year_revenue_growth_rate: str
    """Example: "2-Year Revenue Growth Rate: 12.5%" """
    free_cash_flow_growth: str
    """Example: "Free Cash Flow Growth: 10.2%" """

class ValuationMetrics(BaseModel):
    """
    Represents key valuation metrics for a company.
    """
    market_cap: str
    """Example: "Market Cap: $2.75T" """
    enterprise_value: str
    """Example: "Enterprise Value: $2.65T" """
    trailing_pe_ratio: str
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
    trailing_eps: str
    """Example: "Trailing EPS: $10.25" """
    forward_eps: str
    """Example: "Forward EPS: $12.50" """

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
    cash_dividends_paid: str
    """Example: "Cash Dividends Paid: $14.85B" """
    dividend_yield: str
    """Example: "Dividend Yield: 0.55%" """
    dividend_rate: str
    """Example: "Annual Dividend Rate: $0.96" """
    payout_ratio: str
    """Example: "Payout Ratio: 15.20%" """
    ex_dividend_date: str
    """Example: "Ex-Dividend Date: 2024-03-15" """
    next_dividend_date: str
    """Example: "Next Dividend Date: 2024-06-15" """ 
    

class FinancialData(BaseModel):
    """
    Represents the complete financial data analysis of a company.
    """
    income_statement_analysis: IncomeStatementAnalysis
    balance_sheet_analysis: BalanceSheetAnalysis
    cash_flow_analysis: CashFlowAnalysis
    growth_metrics: GrowthMetrics
    valuation_metrics: ValuationMetrics
    price_metrics: PriceMetrics
    return_metrics: ReturnMetrics
    dividend_metrics: DividendMetrics


#######################
# Risk Analysis Models
#######################

class RiskAnalysisModel(BaseModel):
    """
    Defines the scope and structure for the Risk Analysis section of a stock analysis report.
    
    Example:
    ```json
    {
        "operational_risks": "Supply chain concentration in Asia poses geopolitical risk; 70% of manufacturing in single region.",
        "financial_risks": "High debt levels and interest coverage ratio of 2.5x, exposed to interest rate risk.",
        "market_volatility": "High interest rate sensitivity with beta of 1.2, exposed to tech sector rotation risk.",
        "regulatory_risks": "Antitrust investigations in EU and US markets could impact app store revenue model.",
        "macroeconomic_risks": "Global recession risk, interest rate hikes, and geopolitical tensions.",
        "esg_risks": "Carbon footprint and supply chain sustainability concerns.",
        "company_specific_issues": "Supply chain concentration in Asia poses geopolitical risk; 70% of manufacturing in single region.",
        "other_risks": "Cybersecurity threats and talent retention challenges in competitive labor market."
    }
    ```
    """
    operational_risks: Optional[str] = None
    """Example: "Supply chain concentration in Asia poses geopolitical risk; 70% of manufacturing in single region." """

    financial_risks: Optional[str] = None
    """Example: "High debt levels and interest coverage ratio of 2.5x, exposed to interest rate risk." """
    
    market_volatility: Optional[str] = None
    """Example: "High interest rate sensitivity with beta of 1.2, exposed to tech sector rotation risk." """
    
    regulatory_risks: Optional[str] = None
    """Example: "Antitrust investigations in EU and US markets could impact app store revenue model." """

    macroeconomic_risks: Optional[str] = None
    """Example: "Global recession risk, interest rate hikes, and geopolitical tensions." """

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
    """Example: [
        {"risk": "Market volatility", "severity": 6},
        {"risk": "Regulatory risks", "severity": 5},
        {"risk": "Macroeconomic risks", "severity": 4}
    ]"""


#######################
# News & Research Models
####################### 

class Article(BaseModel):
    """
    Represents a news article.
    """
    title: str
    """Example: "Apple to unveil new iPhone models in September" """
    url: str
    """Example: "https://www.apple.com/newsroom/2024/03/apple-to-unveil-new-iphone-models-in-september/" """
    article_text: str
    """Example: "Apple is expected to unveil new iPhone models in September 2024. The new models are expected to feature improved camera systems and faster processors." """


class NewsAndResearchModel(BaseModel):
    """
    Represents the news and research analysis of a company.
    """
    articles: List[Article]
    """Example: [Article(title="Apple to unveil new iPhone models in September", url="https://www.apple.com/newsroom/2024/03/apple-to-unveil-new-iphone-models-in-september/", article_text="Apple is expected to unveil new iPhone models in September 2024. The new models are expected to feature improved camera systems and faster processors.")] """



######################
# CHART READING MODELS
######################


# # 1. Support & Resistance (Breakout) Chart Model
# class SupportResistanceLevel(BaseModel):
#     """
#     Represents a support or resistance level on a chart.
#     """
#     level_type: Literal["support", "resistance"] = Field(description="Type of level: 'support' or 'resistance'")
#     price: float = Field(description="Price at which the level occurs")
#     date: datetime = Field(description="Date at which the level was identified")

# class BreakoutSignal(BaseModel):
#     """
#     Represents a breakout signal on a support and resistance chart.
#     """
#     direction: str = Field(description="'breakout' or 'breakdown'")
#     price: float = Field(description="Price at which breakout occurs")
#     date: datetime = Field(description="Date at which the breakout occurred")


# class SupportResistanceChart(BaseModel):
#     """
#     Represents a support and resistance chart with breakout signals.
#     """
#     symbol: str = Field(description="Symbol of the stock")
#     method: Literal["Fractal_Pattern", "Window_Shifting"] = Field(description="Method used for level detection")
#     support_levels: List[SupportResistanceLevel] = Field(description="List of support levels")
#     resistance_levels: List[SupportResistanceLevel] = Field(description="List of resistance levels")
#     recent_signals: List[BreakoutSignal] = Field(description="List of recent breakout/breakdown signals")
#     level_count: int = Field(description="Total number of support and resistance levels detected")
#     has_breakout: bool = Field(description="Indicates if a breakout occurred based on the last two candles")
#     last_updated: datetime = Field(description="Last updated date")


# # 2. Squeeze Momentum Indicator Chart Model
# class MomentumBar(BaseModel):
#     latest_value: float = Field(description="Momentum bar value.")
#     latest_color: Literal["lime", "green", "red", "maroon"] = Field(description="Color indicating bullish/bearish momentum")

# class SqueezeStatus(BaseModel):
#     is_squeeze: bool = Field(description="True if currently in a squeeze (black 'x' markers), False otherwise (gray 'x' markers).")
#     current_status: Literal["Squeeze ON (Consolidation)", "Squeeze OFF (Trending)"] = Field(description="Squeeze status")
#     current_momentum: Literal["Increasing Bullish Momentum", "Decreasing Bullish Momentum", "Increasing Bearish Momentum", "Decreasing Bearish Momentum"] = Field(description="Momentum direction")

# class Bands(BaseModel):
#     upper_bb: List[float] = Field(description="List of values for the upper Bollinger Band.")
#     lower_bb: List[float] = Field(description="List of values for the lower Bollinger Band.")
#     upper_kc: List[float] = Field(description="List of values for the upper Keltner Channel.")
#     lower_kc: List[float] = Field(description="List of values for the lower Keltner Channel.")

# class SqueezeMomentumChart(BaseModel):
#     symbol: str = Field(description="Symbol of the stock.")
#     squeeze_status: SqueezeStatus
#     recent_momentum_bars: List[MomentumBar] = Field(description="List of recent momentum bars with values and colors.")
#     bands: Bands = Field(description="Bollinger Bands and Keltner Channel values.")
#     long_signal: bool = Field(description="True if a long signal is detected, False otherwise.")
#     short_signal: bool = Field(description="True if a short signal is detected, False otherwise.")
#     last_updated: datetime = Field(description="Last updated date.")


# # 3. Supertrend Indicator and Backtest Chart Model
# class SupertrendParameters(BaseModel):
#     atr_period: int = Field(description="ATR period used for Supertrend calculation.")
#     atr_multiplier: float = Field(description="ATR multiplier used for Supertrend calculation.")

# class SupertrendIndicator(BaseModel):
#     supertrend: List[bool] = Field(description="List of boolean values indicating the Supertrend direction (True for uptrend, False for downtrend).")
#     final_lowerband: List[float] = Field(description="List of values for the final lower band.")
#     final_upperband: List[float] = Field(description="List of values for the final upper band.")
    
# # class SupertrendTrade(BaseModel):
# #     entry_date: datetime = Field(description="Date of the entry signal.")
# #     entry_price: float = Field(description="Price at the entry signal.")
# #     exit_date: Optional[datetime] = Field(description="Date of the exit signal.")
# #     exit_price: Optional[float] = Field(description="Price at the exit signal.")

# # class SupertrendBacktest(BaseModel):
# #     investment: float = Field(description="Initial investment amount.")
# #     final_equity: float = Field(description="Final equity after backtesting.")
# #     earning: float = Field(description="Earnings from the investment.")
# #     roi_percent: float = Field(description="Return on investment percentage.")
# #     total_trades: int = Field(description="Total number of trades executed.")
# #     trades: List[SupertrendTrade] = Field(description="List of trades executed during backtesting.")

# class SupertrendChart(BaseModel):
#     symbol: str = Field(description="Symbol of the stock.")
#     parameters: SupertrendParameters = Field(description="Parameters used for Supertrend calculation.")
#     indicator_data: SupertrendIndicator = Field(description="Supertrend indicator values.")
#     # backtest_results: SupertrendBacktest = Field(description="Results of the Supertrend backtest.")
#     last_updated: datetime = Field(description="Last updated date.")


# # 4. MACD & Stochastic Oscillator Chart Model
# class MACDStatus(BaseModel):
#     latest_macd_value: float = Field(description="The MACD value.")
#     latest_signal_value: float = Field(description="The signal line value.")
#     latest_histogram_value: float = Field(description="The MACD histogram value (difference between MACD and signal).")
#     current_momentum: Literal["bullish", "bearish", "neutral"] = Field(description="Overall MACD momentum")
#     recent_crossover: Optional[Literal["bullish crossover", "bearish crossover"]] = Field(description="Recent MACD crossover")

# class StochasticStatus(BaseModel):
#     latest_percent_d: float = Field(description="The %D value (fast stochastic).")
#     latest_percent_sd: float = Field(description="The %SD value (slow stochastic).")
#     current_condition: Literal["overbought", "oversold", "neutral"] = Field(description="Stochastic oscillator condition")
#     recent_crossover: Optional[Literal["bullish crossover", "bearish crossover"]] = Field(description="Recent stochastic crossover")

# class MovingAverages(BaseModel):
#     latest_ma_5: Optional[float] = Field(description="5-period moving average value.")
#     latest_ma_20: Optional[float] = Field(description="20-period moving average value.")
#     latest_price_vs_ma_5: str = Field(description="Price relative to 5-period MA: 'above', 'below', or 'crossing'.")
#     latest_price_vs_ma_20: str = Field(description="Price relative to 20-period MA: 'above', 'below', or 'crossing'.")
#     latest_ma_5_vs_ma_20: Optional[str] = Field(description="5-period MA relative to 20-period MA: 'above', 'below', or 'crossing'.")

# class MACDStochasticChart(BaseModel):
#     symbol: str = Field(description="The stock symbol.")
#     macd_status: MACDStatus
#     stochastic_status: StochasticStatus
#     moving_averages: MovingAverages = Field(description="Moving average information.")
#     latest_volume: Optional[int] = Field(description="Trading volume for the most recent period.")
#     last_updated: datetime = Field(description="Timestamp of the last update.")


# # 5. Ichimoku Cloud Chart Model



######
# EMAIL MODELS
######

class Email(BaseModel):
    subject: str = Field(description="The subject of the email.")
    body: str = Field(description="The body of the email.")


