"""
Constants and thresholds used across investor analysis modules.

This module centralizes magic numbers and configurable parameters used in
financial analysis calculations.
"""

from typing import Dict

# ============================================================================
# GENERAL ANALYSIS PARAMETERS
# ============================================================================

# Number of years of historical data to analyze
MIN_PERIODS_FOR_ANALYSIS = 2
PREFERRED_PERIODS_FOR_ANALYSIS = 4
STANDARD_PERIODS_FOR_TREND = 3

# Price history parameters
PRICE_HISTORY_START_DATE = "2022-01-01"
MIN_PRICE_POINTS_FOR_MOMENTUM = 30
MIN_PRICE_POINTS_FOR_VOLATILITY = 10

# News sentiment parameters
DEFAULT_NEWS_ARTICLES_COUNT = 10
DRUCKENMILLER_NEWS_ARTICLES = 7

# ============================================================================
# VALUATION PARAMETERS
# ============================================================================

# DCF Model defaults
class DCFDefaults:
    """Default parameters for Discounted Cash Flow models."""
    GROWTH_RATE = 0.05  # 5% conservative growth
    DISCOUNT_RATE = 0.09  # 9% discount rate
    TERMINAL_MULTIPLE = 12
    PROJECTION_YEARS = 10


class AckmanDCFDefaults:
    """Bill Ackman's DCF parameters (more conservative)."""
    GROWTH_RATE = 0.06  # 6% growth
    DISCOUNT_RATE = 0.10  # 10% discount rate
    TERMINAL_MULTIPLE = 15
    PROJECTION_YEARS = 5


class WoodDCFDefaults:
    """Cathie Wood's DCF parameters (growth-focused)."""
    GROWTH_RATE = 0.20  # 20% high growth assumption
    DISCOUNT_RATE = 0.15  # 15% discount rate for risk
    TERMINAL_MULTIPLE = 25
    PROJECTION_YEARS = 5


# ============================================================================
# FINANCIAL HEALTH THRESHOLDS
# ============================================================================

# Return on Equity (ROE) thresholds
ROE_EXCELLENT = 0.20  # 20%+ excellent
ROE_STRONG = 0.15  # 15%+ strong
ROE_ACCEPTABLE = 0.10  # 10%+ acceptable

# Current Ratio thresholds
CURRENT_RATIO_EXCELLENT = 2.0  # Graham's preferred minimum
CURRENT_RATIO_GOOD = 1.5
CURRENT_RATIO_ACCEPTABLE = 1.0

# Debt ratios
DEBT_TO_EQUITY_CONSERVATIVE = 0.3  # Very low debt
DEBT_TO_EQUITY_MODERATE = 0.7  # Moderate debt
DEBT_TO_EQUITY_HIGH = 1.5  # High debt threshold
DEBT_TO_ASSETS_CONSERVATIVE = 0.5  # 50% threshold

# Margin thresholds
OPERATING_MARGIN_EXCELLENT = 0.20  # 20%+
OPERATING_MARGIN_STRONG = 0.15  # 15%+
OPERATING_MARGIN_ACCEPTABLE = 0.10  # 10%+

GROSS_MARGIN_EXCELLENT = 0.50  # 50%+
GROSS_MARGIN_GOOD = 0.30  # 30%+

# ============================================================================
# GROWTH THRESHOLDS
# ============================================================================

# Revenue growth rates (cumulative)
REVENUE_GROWTH_EXCEPTIONAL = 1.0  # 100%+ exceptional
REVENUE_GROWTH_STRONG = 0.5  # 50%+ strong
REVENUE_GROWTH_GOOD = 0.4  # 40%+ good
REVENUE_GROWTH_MODERATE = 0.3  # 30%+ moderate
REVENUE_GROWTH_SLIGHT = 0.15  # 15%+ slight
REVENUE_GROWTH_MINIMAL = 0.05  # 5%+ minimal

# EPS growth rates
EPS_GROWTH_EXCEPTIONAL = 0.50  # 50%+ exceptional
EPS_GROWTH_GOOD = 0.30  # 30%+ good
EPS_GROWTH_STRONG = 0.30  # 30%+
EPS_GROWTH_MODERATE = 0.15  # 15%+
EPS_GROWTH_SLIGHT = 0.05  # 5%+

# R&D intensity (R&D / Revenue)
RD_INTENSITY_HIGH = 0.15  # 15%+ high innovation investment
RD_INTENSITY_MODERATE = 0.08  # 8%+ moderate
RD_INTENSITY_LOW = 0.05  # 5%+ some investment

# ============================================================================
# VALUATION MULTIPLES
# ============================================================================

# P/E Ratio thresholds
PE_ATTRACTIVE = 15  # <15 attractive
PE_FAIR = 25  # <25 fair
# >25 expensive

# P/FCF Ratio thresholds
PFCF_ATTRACTIVE = 15
PFCF_FAIR = 25

# EV/EBIT thresholds
EV_EBIT_ATTRACTIVE = 15
EV_EBIT_FAIR = 25

# EV/EBITDA thresholds
EV_EBITDA_ATTRACTIVE = 10
EV_EBITDA_FAIR = 18

# FCF Yield thresholds
FCF_YIELD_EXCELLENT = 0.08  # 8%+
FCF_YIELD_GOOD = 0.05  # 5%+
FCF_YIELD_FAIR = 0.03  # 3%+

# ============================================================================
# MOMENTUM & TECHNICAL THRESHOLDS
# ============================================================================

# Price momentum thresholds
MOMENTUM_VERY_STRONG = 0.5  # 50%+ gain
MOMENTUM_MODERATE = 0.2  # 20%+ gain
MOMENTUM_SLIGHT = 0.0  # Any positive gain

# Volatility thresholds (daily returns standard deviation)
VOLATILITY_LOW = 0.01  # 1% daily stdev
VOLATILITY_MODERATE = 0.02  # 2% daily stdev
VOLATILITY_HIGH = 0.04  # 4% daily stdev

# ============================================================================
# MARGIN OF SAFETY THRESHOLDS
# ============================================================================

MARGIN_OF_SAFETY_SIGNIFICANT = 0.50  # 50%+ discount to intrinsic value
MARGIN_OF_SAFETY_LARGE = 0.50  # 50%+ large margin
MARGIN_OF_SAFETY_GOOD = 0.30  # 30%+ discount
MARGIN_OF_SAFETY_MODERATE = 0.20  # 20%+ moderate margin
MARGIN_OF_SAFETY_ACCEPTABLE = 0.10  # 10%+ discount

# Graham-specific
GRAHAM_MARGIN_STRONG = 0.50  # 50%+
GRAHAM_MARGIN_MODERATE = 0.20  # 20%+

# ============================================================================
# QUALITY METRICS
# ============================================================================

# ROIC (Return on Invested Capital) - Munger's favorite
ROIC_EXCELLENT = 0.15  # 15%+ indicates moat
ROIC_GOOD = 0.12  # 12%+ good return

# Free Cash Flow conversion
FCF_TO_NI_EXCELLENT = 1.1  # FCF > Net Income (good accounting)
FCF_TO_NI_GOOD = 0.9  # FCF ≈ Net Income
FCF_TO_NI_MODERATE = 0.7  # FCF somewhat lower

# Cash management (Cash / Revenue)
CASH_TO_REVENUE_OPTIMAL_MIN = 0.10  # 10% minimum
CASH_TO_REVENUE_OPTIMAL_MAX = 0.25  # 25% maximum
CASH_TO_REVENUE_MIN_IDEAL = 0.10  # 10% minimum ideal
CASH_TO_REVENUE_MAX_IDEAL = 0.25  # 25% maximum ideal
CASH_TO_REVENUE_ACCEPTABLE_MIN = 0.05  # 5% minimum
CASH_TO_REVENUE_ACCEPTABLE_MAX = 0.40  # 40% maximum

# Capital expenditure intensity (CapEx / Revenue)
CAPEX_TO_REVENUE_LOW = 0.05  # <5% low capital requirements
CAPEX_TO_REVENUE_MODERATE = 0.10  # <10% moderate
CAPEX_INTENSITY_LOW = 0.05  # <5% low capital requirements
CAPEX_INTENSITY_MODERATE = 0.10  # <10% moderate

# ============================================================================
# CONSISTENCY & PREDICTABILITY THRESHOLDS
# ============================================================================

# Revenue growth volatility (average deviation from mean)
REVENUE_VOLATILITY_LOW = 0.10  # 10% deviation - highly predictable
REVENUE_VOLATILITY_MODERATE = 0.20  # 20% deviation - moderately predictable

# Margin volatility
MARGIN_VOLATILITY_VERY_LOW = 0.03  # 3% - very stable
MARGIN_VOLATILITY_MODERATE = 0.07  # 7% - moderately stable

# Minimum percentage of periods with positive metric
CONSISTENCY_THRESHOLD_EXCELLENT = 1.0  # 100% of periods
CONSISTENCY_THRESHOLD_GOOD = 0.80  # 80% of periods
CONSISTENCY_THRESHOLD_ACCEPTABLE = 0.75  # 75% of periods
CONSISTENCY_THRESHOLD_MODERATE = 0.60  # 60% of periods
CONSISTENCY_THRESHOLD_MAJORITY = 0.50  # 50%+ of periods

# ============================================================================
# INSIDER ACTIVITY THRESHOLDS
# ============================================================================

INSIDER_BUYING_HEAVY = 0.7  # 70%+ of transactions are buys
INSIDER_BUYING_MODERATE = 0.4  # 40%+ buys
INSIDER_SELLING_HEAVY_MIN_SELLS = 5  # Minimum sells to be concerning
INSIDER_SELLING_THRESHOLD = 0.1  # <10% buys is concerning

# ============================================================================
# SHARE COUNT CHANGES
# ============================================================================

SHARE_REDUCTION_SIGNIFICANT = 0.95  # 5%+ reduction (buybacks)
SHARE_STABLE_MAX = 1.05  # <5% increase (stable)
SHARE_DILUTION_SIGNIFICANT = 1.20  # 20%+ increase (concerning)

# ============================================================================
# CATHIE WOOD SPECIFIC (INNOVATION FOCUS)
# ============================================================================

# Disruptive sectors and industries
DISRUPTIVE_SECTORS = [
    "technology",
    "healthcare",
    "communication-services",
    "consumer-cyclical",
    "energy",
    "industrials",
]

DISRUPTIVE_INDUSTRIES = [
    # Genomics
    "biotechnology",
    "diagnostics-research",
    "health-information-services",
    "medical-devices",
    # Robotics & AI
    "semiconductors",
    "software-application",
    "software-infrastructure",
    "electronic-components",
    "information-technology-services",
    "communication-equipment",
    "computer-hardware",
    "aerospace-defense",
    "specialty-industrial-machinery",
    # Energy Storage
    "solar",
    "utilities-renewable",
    "oil-gas-e-p",
    # Blockchain
    "financial-data-stock-exchanges",
    # Next Gen Internet
    "internet-content-information",
    "electronic-gaming-multimedia",
]

# ============================================================================
# SCORING WEIGHTS
# ============================================================================

# Druckenmiller scoring weights
DRUCKENMILLER_WEIGHTS: Dict[str, float] = {
    "growth_momentum": 0.35,
    "risk_reward": 0.20,
    "valuation": 0.20,
    "sentiment": 0.15,
    "insider_activity": 0.10,
}

# Signal thresholds (out of 10 or percentage of max score)
SIGNAL_BULLISH_THRESHOLD = 0.70  # 70%+ of max score
SIGNAL_BEARISH_THRESHOLD = 0.30  # 30% or less of max score
# Between = neutral

# Druckenmiller specific signal thresholds (out of 10)
DRUCKENMILLER_BULLISH = 7.5
DRUCKENMILLER_BEARISH = 4.5

# ============================================================================
# NEWS SENTIMENT THRESHOLDS
# ============================================================================

SENTIMENT_VERY_NEGATIVE = -30  # Highly bearish
SENTIMENT_NEGATIVE = 0  # Some negativity
SENTIMENT_VERY_POSITIVE = 30  # Highly bullish
# Otherwise neutral/positive
