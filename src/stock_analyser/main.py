import os
import time
import json
import random
import yfinance as yf
from pathlib import Path
from textwrap import dedent
from datetime import datetime

from .plotting.plot_risk_severity import plot_risk_severity
from .plotting.plot_dcf_sensitivity import plot_dcf_sensitivity
from .plotting.plot_financial_metric_trends import (
    plot_net_income_trends,
    plot_revenue_trends,
    plot_gross_margin_trends,
    plot_earnings_per_share_trends,
    plot_free_cash_flow_trends,
)
from .plotting.plot_macd_stochastic import plot_macd_stochastic
from .plotting.plot_relative_strength import plot_relative_strength
from .plotting.plot_squeeze_momentum import analyze_single_stock
from .plotting.plot_competitor_prices import plot_competitor_prices
from .plotting.plotting_tools import plot_candlestick_chart
from .plotting.plot_supertrend_chart import analyze_single_stock as analyze_supertrend
from .plotting.plot_standard_error_bands import plot_standard_error_bands
from .plotting.plot_multi_timeframe_momentum import plot_multi_timeframe_momentum, get_momentum_consensus

from .utils.get_sp500_tickers import get_sp500_tickers
from .utils.setup import Setup
from .utils.models import ReportState
from .utils.decorators import timeit
from .utils.convert_to_pdf import convert_md_to_pdf
from .utils.prompts import (
    critic_guidelines,
    editor_guidelines,
    expert_analyst_writer_prompt,
    writer_guidelines,
)
from .utils.constants import TIMESTAMP, OUTPUT_DIR, PLOTS_DIR, BACKOFF_TIME

from crewai.flow.flow import Flow, listen, start, and_
from .crews.news_and_research_crew.news_and_research_crew import NewsAndResearchCrew
from .crews.market_and_industry_crew.market_and_industry_crew import MarketAndIndustryCrew
from .crews.competition_identifier_crew.competition_identifier_crew import CompetitionIdentifierCrew
from .crews.competitor_crew.competitor_crew import CompetitorCrew
from .crews.financial_data_crew.financial_data_crew import FinancialDataCrew
from .crews.trends_crew.trends_crew import TrendsCrew
from .crews.analyst_insights_crew.analyst_insights_crew import AnalystInsightsCrew
from .crews.benjamin_graham_crew.benjamin_graham_crew import BenjaminGrahamCrew
from .crews.warren_buffet_crew.warren_buffet_crew import WarrenBuffetCrew
from .crews.charlie_munger_crew.charlie_munger_crew import CharlieMungerCrew
from .crews.cathie_wood_crew.cathie_wood_crew import CathieWoodCrew
from .crews.stanley_druckenmiller_crew.stanley_druckenmiller_crew import StanleyDruckenmillerCrew
from .crews.bill_ackman_crew.bill_ackman_crew import BillAckmanCrew
from .crews.risk_analysis_crew.risk_analysis_crew import RiskAnalysisCrew
from .crews.risk_severity_crew.risk_severity_crew import RiskSeverityCrew
from .crews.investment_recommendation_crew.investment_recommendation_crew import InvestmentRecommendationCrew
from .crews.future_outlook_crew.future_outlook_crew import FutureOutlookCrew
from .crews.executive_summary_crew.executive_summary_crew import ExecutiveSummaryCrew
from .crews.conclusion_crew.conclusion_crew import ConclusionCrew
from .crews.gmail_attachment_crew.gmail_attachment_crew import GmailAttachmentCrew

beginning_time = time.time()


class ReportFlow(Flow[ReportState]):
    knowledge_source = None
    setup_instance = None

    @timeit
    @start()
    def confirm_company_stock(self):
        while True:
            if not self.state.company_ticker:
                self.state.company_ticker = (
                    input(
                        "\n\033[1;34mEnter the stock ticker symbol of the company you want to analyze or hit 'ENTER' for a random S&P 500 company:\033[0m "
                    )
                    .strip()
                    .upper()
                )
            if self.state.company_ticker == "":
                while True:
                    sp500_tickers = get_sp500_tickers()
                    self.state.company_ticker = random.choice(sp500_tickers)
                    history = yf.Ticker(self.state.company_ticker).history(
                        period="7d", interval="1d"
                    )
                    if not history.empty:
                        break
                    else:
                        print(
                            f"❌ Invalid ticker: {self.state.company_ticker}. Removing it from the list."
                        )
                        sp500_tickers.remove(self.state.company_ticker)
            history = yf.Ticker(self.state.company_ticker).history(
                period="7d", interval="1d"
            )
            if not history.empty:
                company = yf.Ticker(self.state.company_ticker)

                self.state.company_name = (
                    company.info["displayName"]
                    if "displayName" in company.info
                    else company.info["longName"]
                    if "longName" in company.info
                    else company.info["shortName"]
                    if "shortName" in company.info
                    else company.info["longBusinessSummary"]
                    if "longBusinessSummary" in company.info
                    else "Unknown company name"
                )
                self.state.industry = (
                    company.info["industry"]
                    if "industry" in company.info
                    else "Unknown industry"
                )
                self.state.company_website = (
                    company.info["website"]
                    if "website" in company.info
                    else "Unknown website"
                )

                print(f"Ticker: \033[1;37m{self.state.company_ticker}\033[0m")
                print(f"Company name: \033[1;37m{self.state.company_name}\033[0m")
                print(f"Industry: \033[1;37m{self.state.industry}\033[0m")
                print(f"Website: \033[1;37m{self.state.company_website}\033[0m\n")
                break
            else:
                print("❌ Invalid ticker. Please try again.\n")
                self.state.company_ticker = ""

    @timeit
    @listen(confirm_company_stock)
    def set_up(self):
        # Create a Setup instance and store it as a class variable
        setup = Setup(self.state)

        # Store the setup instance in the class variable
        ReportFlow.setup_instance = setup

        # Initialize analysis
        setup_success = setup.initialize_analysis()

        if setup_success:
            print("Analysis initialization completed successfully ✅")
        else:
            print("⚠️ Failed to initialize analysis.")
            print(
                "⚠️ Please check QDRANT_CLUSTER_URL, QDRANT_API_KEY, and GEMINI_API_KEY environment variables."
            )
            print("⚠️ Also verify that the Qdrant server is running and accessible.")

    @timeit
    @listen(set_up)
    def generate_news_and_research_section(self):
        print("Generating news and research section...")

        inputs = {
            "company_name": self.state.company_name,
            "company_ticker": self.state.company_ticker,
            "date_iso": self.state.date_iso,
            "date_us": self.state.date_us,
            "industry": self.state.industry,
            "writer_guidelines": writer_guidelines,
        }

        news_and_research_section = NewsAndResearchCrew().crew().kickoff(inputs=inputs)

        self.state.news_and_research_section = news_and_research_section.raw
        print("News and research section generated ✅")
        print(
            f"🪙 Number of tokens used: {news_and_research_section.token_usage.total_tokens:,}"
        )
        self.state.total_token_usage += (
            news_and_research_section.token_usage.total_tokens
        )

    @timeit
    @listen(generate_news_and_research_section)
    def generate_market_and_industry_section(self):
        print("Generating market and industry section...")

        inputs = {
            "company_name": self.state.company_name,
            "company_ticker": self.state.company_ticker,
            "company_website": self.state.company_website,
            "industry": self.state.industry,
            "date_iso": self.state.date_iso,
            "date_us": self.state.date_us,
            "critic_guidelines": critic_guidelines,
            "editor_guidelines": editor_guidelines,
            "writer_guidelines": writer_guidelines,
        }

        market_and_industry_section = (
            MarketAndIndustryCrew().crew().kickoff(inputs=inputs)
        )

        self.state.market_and_industry_section = market_and_industry_section.raw
        print("Market and industry section generated ✅")
        print(
            f"🪙 Number of tokens used: {market_and_industry_section.token_usage.total_tokens:,}"
        )
        self.state.total_token_usage += (
            market_and_industry_section.token_usage.total_tokens
        )


    @timeit
    @listen(generate_market_and_industry_section)
    def identify_competitors(self):
        print("Identifying competitors...")

        inputs = {
            "company_name": self.state.company_name,
            "company_ticker": self.state.company_ticker,
            "industry": self.state.industry,
            "date_iso": self.state.date_iso,
        }

        competition_identifier_section = CompetitionIdentifierCrew().crew().kickoff(inputs=inputs)

        self.state.competitors_list = competition_identifier_section.pydantic.competitors
        print("Competition identifier section generated ✅")

    
    @timeit
    @listen(identify_competitors)
    def generate_competitor_landscape_section(self):
        print("Generating competitor landscape section...")

        inputs = {
            "company_ticker": self.state.company_ticker,
            "company_name": self.state.company_name,
            "industry": self.state.industry,
            "date_iso": self.state.date_iso,
            "date_us": self.state.date_us,
            "competitors_list": self.state.competitors_list,
        }

        crew_result = CompetitorCrew().crew().kickoff(inputs=inputs)

        self.state.competitor_landscape_section = crew_result.raw
        print("Competitor landscape section generated ✅")
        print(
            f"🪙 Number of tokens used: {crew_result.token_usage.total_tokens:,}"
        )
        self.state.total_token_usage += crew_result.token_usage.total_tokens


    @timeit
    @listen(generate_competitor_landscape_section)
    def plot_competitor_prices(self):
        print("Plotting competitor prices...")

        try:
            competitor_prices_plot = plot_competitor_prices(
                self.state.competitors_list, PLOTS_DIR, TIMESTAMP
            )
        except Exception as e:
            print(f"❌ Error plotting competitor prices: {str(e)}")

        if competitor_prices_plot:
            print(f"📊 Competitor prices plot saved to {competitor_prices_plot}\n")
            self.state.competitor_landscape_section = self.state.competitor_landscape_section.replace(
                "[competitorPrices]",
                f"\n\n### Competitor Prices\n![Competitor Price Plot]({competitor_prices_plot})\n\n",
            )
        else:
            print("❌ Competitor prices plot not generated. Skipping...")
            self.state.competitor_landscape_section = self.state.competitor_landscape_section.replace(
                "[competitorPrices]",
                ""
            )

    @timeit
    @listen(generate_competitor_landscape_section)
    def generate_financial_data_section(self):
        print("Generating financial data section...")

        inputs = {
            "company_name": self.state.company_name,
            "company_ticker": self.state.company_ticker,
            "company_website": self.state.company_website,
            "industry": self.state.industry,
            "date_iso": self.state.date_iso,
            "date_us": self.state.date_us,
            "critic_guidelines": critic_guidelines,
            "editor_guidelines": editor_guidelines,
            "writer_guidelines": writer_guidelines,
        }

        financial_data_section = FinancialDataCrew().crew().kickoff(inputs=inputs)

        self.state.financial_data_section = financial_data_section.raw
        print("Financial data section generated ✅")
        print(
            f"🪙 Number of tokens used: {financial_data_section.token_usage.total_tokens:,}"
        )
        self.state.total_token_usage += financial_data_section.token_usage.total_tokens

    @timeit
    @listen(generate_financial_data_section)
    def plot_historical_stock_data(self):
        print("Plotting historical stock data...")

        try:
            historical_stock_data_plot = plot_macd_stochastic(
                self.state.company_ticker, "6mo", PLOTS_DIR, TIMESTAMP
            )
            print(
                f"📊 MACD and Stochastic Oscillator chart saved to {historical_stock_data_plot}\n"
            )
        except Exception as e:
            print(f"❌ Error plotting MACD and Stochastic Oscillator chart: {str(e)}")

        if historical_stock_data_plot:
            macd = "## Financial Deep Dive"
            macd += f"\n\n![MACD and Stochastic Oscillator Plot]({historical_stock_data_plot})"
            macd += (
                "<figcaption>This chart combines candlestick price action and volume "
                "with Moving Average Convergence Divergence (MACD) (center panel) and "
                "the Stochastic oscillator (bottom panel) to show market momentum and "
                "identify potential trend reversals. MACD highlights momentum shifts "
                "through histogram bars and signal line crossovers, while the Stochastic "
                "oscillator indicates overbought or oversold conditions through threshold "
                "lines and line crossovers.</figcaption>\n\n"
            )

            self.state.financial_data_section = (
                self.state.financial_data_section.replace(
                    "## Financial Deep Dive", macd
                )
            )
        else:
            print("❌ Historical stock data plot not generated. Skipping...")

    @timeit
    @listen(generate_financial_data_section)
    def generate_trends_section(self):
        print("Generating trends section...")

        # Generate multi-timeframe momentum dashboard
        print("Generating multi-timeframe momentum dashboard...")
        try:
            momentum_plot = plot_multi_timeframe_momentum(
                self.state.company_ticker, PLOTS_DIR, TIMESTAMP
            )
            momentum_consensus = get_momentum_consensus(
                self.state.company_ticker, PLOTS_DIR, TIMESTAMP
            )
            print(f"📊 Multi-timeframe momentum dashboard saved to {momentum_plot}")
        except Exception as e:
            print(f"❌ Error generating momentum dashboard: {str(e)}")
            momentum_plot = None
            momentum_consensus = None

        inputs = {
            "company_name": self.state.company_name,
            "company_ticker": self.state.company_ticker,
            "date_iso": self.state.date_iso,
            "momentum_analysis": momentum_consensus,
        }

        raw_trends_section = TrendsCrew().crew().kickoff(inputs=inputs)

        self.state.trends_section = "### Recent trends\n\n"
        trends_section = raw_trends_section.raw

        # Plot financial metrics trends
        revenue_plot = plot_revenue_trends(
            self.state.company_ticker, PLOTS_DIR, TIMESTAMP
        )
        net_income_plot = plot_net_income_trends(
            self.state.company_ticker, PLOTS_DIR, TIMESTAMP
        )
        gross_margin_plot = plot_gross_margin_trends(
            self.state.company_ticker, PLOTS_DIR, TIMESTAMP
        )
        earnings_per_share_plot = plot_earnings_per_share_trends(
            self.state.company_ticker, PLOTS_DIR, TIMESTAMP
        )
        free_cash_flow_plot = plot_free_cash_flow_trends(
            self.state.company_ticker, PLOTS_DIR, TIMESTAMP
        )

        trends_section = trends_section.replace(
            "[revenue]", f"\n![Revenue Trends Plot]({revenue_plot})\n\n\n"
        )
        trends_section = trends_section.replace(
            "[netIncome]", f"\n![Net Income Trends Plot]({net_income_plot})\n\n\n"
        )
        trends_section = trends_section.replace(
            "[grossMargin]", f"\n![Gross Margin Trends Plot]({gross_margin_plot})\n\n\n"
        )
        trends_section = trends_section.replace(
            "[eps]",
            f"\n![Earnings Per Share Trends Plot]({earnings_per_share_plot})\n\n\n",
        )
        trends_section = trends_section.replace(
            "[freeCashFlow]",
            f"\n![Free Cash Flow Trends Plot]({free_cash_flow_plot})\n\n\n",
        )
        
        # Add multi-timeframe momentum dashboard
        if momentum_plot:
            trends_section = trends_section.replace(
                "[multiTimeframeMomentum]", 
                f"\n![Multi Timeframe Momentum Dashboard]({momentum_plot})\n\n"
                + "<figcaption>Multi-timeframe momentum analysis comparing key technical "
                + "indicators (RSI, Williams %R, Ultimate Oscillator, Schaff Trend Cycle) "
                + "across daily, weekly, and monthly timeframes. This dashboard provides "
                + "comprehensive momentum confirmation and helps identify trend alignment "
                + "or divergence across different time horizons for better trend "
                + "validation.</figcaption>\n\n\n"
            )
        else:
            trends_section = trends_section.replace("[multiTimeframeMomentum]", "")

        self.state.trends_section += trends_section

        print("Trends section generated ✅")
        print(
            f"🪙 Number of tokens used: {raw_trends_section.token_usage.total_tokens:,}"
        )
        self.state.total_token_usage += raw_trends_section.token_usage.total_tokens

    @timeit
    @listen(generate_trends_section)
    def generate_analyst_insights_section(self):
        print("Generating analyst insights section...")

        inputs = {
            "company_name": self.state.company_name,
            "company_ticker": self.state.company_ticker,
            "company_website": self.state.company_website,
            "industry": self.state.industry,
            "date_iso": self.state.date_iso,
            "date_us": self.state.date_us,
            "critic_guidelines": critic_guidelines,
            "editor_guidelines": editor_guidelines,
            "writer_guidelines": writer_guidelines,
        }

        analyst_insights_section = AnalystInsightsCrew().crew().kickoff(inputs=inputs)

        self.state.analyst_insights_section = analyst_insights_section.raw
        print("Analyst insights section generated ✅")
        print(
            f"🪙 Number of tokens used: {analyst_insights_section.token_usage.total_tokens:,}"
        )
        self.state.total_token_usage += (
            analyst_insights_section.token_usage.total_tokens
        )

    @timeit
    @listen(generate_analyst_insights_section)
    def plot_signals_charts(self):
        print("Plotting signals charts...")

        try:
            rsi_chart = plot_relative_strength(
                self.state.company_ticker, PLOTS_DIR, TIMESTAMP
            )
            if rsi_chart:
                print(f"📊 RSI chart saved to {rsi_chart}\n")
                fig_and_caption = f"\n\n![RSI Chart]({rsi_chart})"
                fig_and_caption += (
                    "\n<figcaption>The top panel shows closing prices with moving "
                    "averages (10-, 20-, 50-, 100-day) to identify key trends; the "
                    "second panel shows daily trading volume for activity insights; the "
                    "third panel compares stock performance to the S&P 500 (RS line), "
                    "highlighting periods of relative strength or weakness relative to a "
                    "10-day moving average; and the bottom panel presents the Relative "
                    "Strength Index (RSI), marking overbought (>70) and oversold (<30) "
                    "conditions for potential trend reversals or continuations.</figcaption>\n\n"
                )
                self.state.analyst_insights_section = (
                    self.state.analyst_insights_section.replace(
                        "[rsiChart]", fig_and_caption
                    )
                )
                print("RSI chart added to analyst insights section ✅")
            else:
                print("❌ RSI chart not added to analyst insights section. Skipping...")
                self.state.analyst_insights_section = (
                    self.state.analyst_insights_section.replace("[rsiChart]", "")
                )
        except Exception as e:
            print(f"❌ Error plotting RSI chart: {str(e)}")

        try:
            squeeze_momentum_signal = analyze_single_stock(
                self.state.company_ticker, output_dir=PLOTS_DIR, timestamp=TIMESTAMP
            )
            squeeze_plot_path = squeeze_momentum_signal.get("plot_path")
            momentum = squeeze_momentum_signal.get("signals", {}).get("momentum", "")
            status = squeeze_momentum_signal.get("signals", {}).get("status", "")
            if squeeze_plot_path:
                print(f"📊 Squeeze Momentum chart saved to {squeeze_plot_path}\n")

                fig_and_caption = f"\n\n![Squeeze Momentum Chart]({squeeze_plot_path})"
                fig_and_caption += (
                    "\n<figcaption>The chart integrates candlestick price action with "
                    "volatility indicators (Bollinger Bands and Keltner Channels) to "
                    'highlight volatility compression ("squeeze") and expansions. '
                    "Colored bars depict bullish (lime/green) or bearish (red/maroon) "
                    "momentum strength, while crosses show squeeze status (black: "
                    "consolidation; gray: trending). Volume bars at the bottom "
                    "contextualize market participation, aiding identification of "
                    "potential breakout or reversal opportunities.\n\n"
                    f"<strong>{momentum}</strong>, {status}</figcaption>\n\n"
                )
                self.state.analyst_insights_section = (
                    self.state.analyst_insights_section.replace(
                        "[squeezeMomentumChart]", fig_and_caption
                    )
                )

                print("Squeeze Momentum chart added to analyst insights section ✅")
            else:
                print(
                    "❌ Squeeze Momentum chart not added to analyst insights section. "
                    "Skipping..."
                )
                self.state.analyst_insights_section = (
                    self.state.analyst_insights_section.replace(
                        "[squeezeMomentumChart]", ""
                    )
                )
        except Exception as e:
            print(f"❌ Error plotting Squeeze Momentum chart: {str(e)}")

        # Plot Supertrend analysis
        try:
            supertrend_result = analyze_supertrend(
                self.state.company_ticker, atr_period=10, atr_multiplier=3.0, 
                start_date='2023-01-01', output_dir=PLOTS_DIR, timestamp=TIMESTAMP
            )
            if supertrend_result:
                _, _, _, roi = supertrend_result
                supertrend_plot_path = f"{PLOTS_DIR}/{self.state.company_ticker}_supertrend_period10_mult3.0_{TIMESTAMP}.png"
                
                if os.path.exists(supertrend_plot_path):
                    print(f"📊 Supertrend chart saved to {supertrend_plot_path}\n")
                    
                    fig_and_caption = f"\n\n![Supertrend Chart]({supertrend_plot_path})"
                    fig_and_caption += (
                        "\n<figcaption>The Supertrend indicator combines price action with "
                        "Average True Range (ATR) to identify trend direction and potential "
                        "reversal points. Green areas indicate bullish trends (buy signals) "
                        "while red areas show bearish trends (sell signals). The indicator "
                        "adapts to market volatility and provides dynamic support and resistance "
                        f"levels. Backtest ROI: {roi:.2f}%</figcaption>\n\n"
                    )
                    
                    self.state.analyst_insights_section = (
                        self.state.analyst_insights_section.replace(
                            "[supertrendChart]", fig_and_caption
                        )
                    )
                    print("Supertrend chart added to analyst insights section ✅")
                else:
                    print("❌ Supertrend chart not found. Skipping...")
                    self.state.analyst_insights_section = (
                        self.state.analyst_insights_section.replace(
                            "[supertrendChart]", ""
                        )
                    )
            else:
                print("❌ Supertrend analysis failed. Skipping...")
                self.state.analyst_insights_section = (
                    self.state.analyst_insights_section.replace(
                        "[supertrendChart]", ""
                    )
                )
        except Exception as e:
            print(f"❌ Error plotting Supertrend chart: {str(e)}")
            self.state.analyst_insights_section = (
                self.state.analyst_insights_section.replace(
                    "[supertrendChart]", ""
                )
            )

        # Plot Standard Error Bands analysis
        try:
            seb_plot_path = plot_standard_error_bands(
                self.state.company_ticker, period='6mo', 
                output_dir=PLOTS_DIR, timestamp=TIMESTAMP
            )
            
            if seb_plot_path and os.path.exists(seb_plot_path):
                print(f"📊 Standard Error Bands chart saved to {seb_plot_path}\n")
                
                fig_and_caption = f"\n\n![Standard Error Bands Chart]({seb_plot_path})"
                fig_and_caption += (
                    "\n<figcaption>Standard Error Bands provide a statistically-based "
                    "approach to volatility analysis using linear regression. The center "
                    "line represents a smoothed regression trend, while the upper and lower "
                    "bands are positioned at ±2 standard errors from the regression line. "
                    "This creates a dynamic channel that adapts to price volatility and "
                    "helps identify potential support/resistance levels and trend "
                    "continuation or reversal points.</figcaption>\n\n"
                )
                
                self.state.analyst_insights_section = (
                    self.state.analyst_insights_section.replace(
                        "[standardErrorBandsChart]", fig_and_caption
                    )
                )
                print("Standard Error Bands chart added to analyst insights section ✅")
            else:
                print("❌ Standard Error Bands chart not generated. Skipping...")
                self.state.analyst_insights_section = (
                    self.state.analyst_insights_section.replace(
                        "[standardErrorBandsChart]", ""
                    )
                )
        except Exception as e:
            print(f"❌ Error plotting Standard Error Bands chart: {str(e)}")
            self.state.analyst_insights_section = (
                self.state.analyst_insights_section.replace(
                    "[standardErrorBandsChart]", ""
                )
            )

    @timeit
    @listen(plot_signals_charts)
    def plot_dcf_sensitivity_3D_chart(self):
        print("Plotting DCF sensitivity 3D surface chart...")

        try:
            dcf_sensitivity_chart = plot_dcf_sensitivity(
                self.state.company_ticker, PLOTS_DIR, TIMESTAMP
            )
            print(f"📊 DCF Sensitivity plot saved to {dcf_sensitivity_chart}\n")

            if dcf_sensitivity_chart:
                chart = f"\n\n![DCF Sensitivity plot]({dcf_sensitivity_chart})"
                chart += (
                    "\n<figcaption>The 3D surface plot illustrates sensitivity "
                    "analysis of implied share price against varying assumptions for "
                    "the Weighted Average Cost of Capital (WACC) and Terminal Growth "
                    "Rate (TGR). The horizontal axes represent different scenarios "
                    "of WACC and TGR, respectively, while the vertical axis shows "
                    "the resulting implied share price derived from discounted cash "
                    "flow (DCF) valuation. The surface demonstrates how sensitive "
                    "the company's valuation is to these two critical parameters; "
                    "areas of steeper slope indicate greater sensitivity, helping "
                    "identify scenarios where valuation is highly dependent on small "
                    "changes in underlying assumptions.</figcaption>\n\n"
                )

                self.state.analyst_insights_section = (
                    self.state.analyst_insights_section.replace(
                        "[sensitivityAnalysis]", chart
                    )
                )
            else:
                print(
                    "❌ DCF Sensitivity plot not added to analyst insights section. Skipping..."
                )
                self.state.analyst_insights_section = (
                    self.state.analyst_insights_section.replace(
                        "[sensitivityAnalysis]", ""
                    )
                )
        except Exception as e:
            print(f"❌ Error plotting DCF Sensitivity plot: {str(e)}")

    @timeit
    @listen(plot_dcf_sensitivity_3D_chart)
    def generate_warren_buffet_section(self):
        print("Generating Warren Buffet section...")

        inputs = {
            "company_name": self.state.company_name,
            "company_ticker": self.state.company_ticker,
            "date_us": self.state.date_us,
            "expert_analyst_writer_prompt": expert_analyst_writer_prompt,
        }

        warren_buffet_section = WarrenBuffetCrew().crew().kickoff(inputs=inputs)

        self.state.warren_buffet_section = warren_buffet_section.raw
        print("Warren Buffet section generated ✅")
        print(
            f"🪙 Number of tokens used: {warren_buffet_section.token_usage.total_tokens:,}"
        )
        self.state.total_token_usage += warren_buffet_section.token_usage.total_tokens

    @timeit
    @listen(generate_warren_buffet_section)
    def generate_cathie_wood_section(self):
        print("Generating Cathie Wood section...")

        inputs = {
            "company_name": self.state.company_name,
            "company_ticker": self.state.company_ticker,
            "date_us": self.state.date_us,
            "expert_analyst_writer_prompt": expert_analyst_writer_prompt,
        }

        cathie_wood_section = CathieWoodCrew().crew().kickoff(inputs=inputs)

        self.state.cathie_wood_section = cathie_wood_section.raw
        print("Cathie Wood section generated ✅")
        print(
            f"🪙 Number of tokens used: {cathie_wood_section.token_usage.total_tokens:,}"
        )
        self.state.total_token_usage += cathie_wood_section.token_usage.total_tokens

        time.sleep(BACKOFF_TIME)
        for remaining in range(BACKOFF_TIME, 0, -1):
            print(
                f"⏳ Pausing for {remaining} seconds to avoid rate limit...", end="\r"
            )
            time.sleep(1)
        print("Resuming...")

    @timeit
    @listen(generate_cathie_wood_section)
    def generate_stanley_druckenmiller_section(self):
        print("Generating Stanley Druckenmiller section...")

        inputs = {
            "company_name": self.state.company_name,
            "company_ticker": self.state.company_ticker,
            "date_us": self.state.date_us,
            "expert_analyst_writer_prompt": expert_analyst_writer_prompt,
        }

        stanley_druckenmiller_section = (
            StanleyDruckenmillerCrew().crew().kickoff(inputs=inputs)
        )

        self.state.stanley_druckenmiller_section = stanley_druckenmiller_section.raw
        print("Stanley Druckenmiller section generated ✅")
        print(
            f"🪙 Number of tokens used: {stanley_druckenmiller_section.token_usage.total_tokens:,}"
        )
        self.state.total_token_usage += (
            stanley_druckenmiller_section.token_usage.total_tokens
        )

    @timeit
    @listen(generate_stanley_druckenmiller_section)
    def generate_benjamin_graham_section(self):
        print("Generating Benjamin Graham section...")

        inputs = {
            "company_name": self.state.company_name,
            "company_ticker": self.state.company_ticker,
            "date_us": self.state.date_us,
            "expert_analyst_writer_prompt": expert_analyst_writer_prompt,
        }

        benjamin_graham_section = BenjaminGrahamCrew().crew().kickoff(inputs=inputs)

        self.state.benjamin_graham_section = benjamin_graham_section.raw
        print("Benjamin Graham section generated ✅")
        print(
            f"🪙 Number of tokens used: {benjamin_graham_section.token_usage.total_tokens:,}"
        )
        self.state.total_token_usage += benjamin_graham_section.token_usage.total_tokens

        time.sleep(BACKOFF_TIME)
        for remaining in range(BACKOFF_TIME, 0, -1):
            print(
                f"⏳ Pausing for {remaining} seconds to avoid rate limit...", end="\r"
            )
            time.sleep(1)
        print("Resuming...")

    @timeit
    @listen(generate_benjamin_graham_section)
    def generate_charlie_munger_section(self):
        print("Generating Charlie Munger section...")

        inputs = {
            "company_name": self.state.company_name,
            "company_ticker": self.state.company_ticker,
            "date_us": self.state.date_us,
            "expert_analyst_writer_prompt": expert_analyst_writer_prompt,
        }

        charlie_munger_section = CharlieMungerCrew().crew().kickoff(inputs=inputs)

        self.state.charlie_munger_section = charlie_munger_section.raw
        print("Charlie Munger section generated ✅")
        print(
            f"🪙 Number of tokens used: {charlie_munger_section.token_usage.total_tokens:,}"
        )
        self.state.total_token_usage += charlie_munger_section.token_usage.total_tokens

        time.sleep(BACKOFF_TIME)
        for remaining in range(BACKOFF_TIME, 0, -1):
            print(
                f"⏳ Pausing for {remaining} seconds to avoid rate limit...", end="\r"
            )
            time.sleep(1)
        print("Resuming...")

    @timeit
    @listen(generate_charlie_munger_section)
    def generate_bill_ackman_section(self):
        print("Generating Bill Ackman section...")

        inputs = {
            "company_name": self.state.company_name,
            "company_ticker": self.state.company_ticker,
            "date_us": self.state.date_us,
            "expert_analyst_writer_prompt": expert_analyst_writer_prompt,
        }

        bill_ackman_section = BillAckmanCrew().crew().kickoff(inputs=inputs)

        self.state.bill_ackman_section = bill_ackman_section.raw
        print("Bill Ackman section generated ✅")
        print(
            f"🪙 Number of tokens used: {bill_ackman_section.token_usage.total_tokens:,}"
        )
        self.state.total_token_usage += bill_ackman_section.token_usage.total_tokens

    @timeit
    @listen(generate_bill_ackman_section)
    def generate_risk_analysis_section(self):
        print("Generating risk analysis section...")

        inputs = {
            "company_name": self.state.company_name,
            "company_ticker": self.state.company_ticker,
            "company_website": self.state.company_website,
            "industry": self.state.industry,
            "date_iso": self.state.date_iso,
            "date_us": self.state.date_us,
            "critic_guidelines": critic_guidelines,
            "editor_guidelines": editor_guidelines,
            "writer_guidelines": writer_guidelines,
        }

        risk_analysis_section = RiskAnalysisCrew().crew().kickoff(inputs=inputs)

        self.state.risk_analysis_section = risk_analysis_section.raw
        self.state.risk_analysis_section += "\n"
        print("Risk analysis section generated ✅")
        print(
            f"🪙 Number of tokens used: {risk_analysis_section.token_usage.total_tokens:,}"
        )
        self.state.total_token_usage += risk_analysis_section.token_usage.total_tokens

    @timeit
    @listen(generate_risk_analysis_section)
    def plot_risk_severity(self):
        print("Plotting risk severity...")

        inputs = {
            "company_name": self.state.company_name,
            "risk_analysis": self.state.risk_analysis_section,
            "date_iso": self.state.date_iso,
        }

        identified_risks = RiskSeverityCrew().crew().kickoff(inputs=inputs).pydantic

        risks = identified_risks.risk_severity_list

        try:
            risk_severity_plot = plot_risk_severity(risks, PLOTS_DIR, TIMESTAMP)
            print(f"Risk severity plot saved to {risk_severity_plot}\n")

            self.state.risk_analysis_section += (
                f"![Risk Severity Plot]({risk_severity_plot})"
            )
        except Exception as e:
            print(f"❌ Error plotting risk severity: {str(e)}")

    @timeit
    @listen(generate_risk_analysis_section)
    def generate_future_outlook_section(self):
        print("Generating future outlook section...")

        inputs = {
            "company_name": self.state.company_name,
            "company_ticker": self.state.company_ticker,
            "company_website": self.state.company_website,
            "industry": self.state.industry,
            "date_iso": self.state.date_iso,
            "date_us": self.state.date_us,
            "critic_guidelines": critic_guidelines,
            "editor_guidelines": editor_guidelines,
            "writer_guidelines": writer_guidelines,
        }

        future_outlook_section = FutureOutlookCrew().crew().kickoff(inputs=inputs)

        self.state.future_outlook_section = future_outlook_section.raw
        print("Future outlook section generated ✅")
        print(
            f"🪙 Number of tokens used: {future_outlook_section.token_usage.total_tokens:,}"
        )
        self.state.total_token_usage += future_outlook_section.token_usage.total_tokens

    @timeit
    @listen(generate_future_outlook_section)
    def generate_investment_recommendations_section(self):
        print("Generating investment recommendations section...")

        inputs = {
            "company_name": self.state.company_name,
            "company_ticker": self.state.company_ticker,
            "company_website": self.state.company_website,
            "industry": self.state.industry,
            "date_iso": self.state.date_iso,
            "date_us": self.state.date_us,
            "market_and_industry_section": self.state.market_and_industry_section,
            "competitor_landscape_section": self.state.competitor_landscape_section,
            "financial_data_section": self.state.financial_data_section,
            "trends_section": self.state.trends_section,
            "analyst_insights_section": self.state.analyst_insights_section,
            "warren_buffet_section": self.state.warren_buffet_section,
            "cathie_wood_section": self.state.cathie_wood_section,
            "stanley_druckenmiller_section": self.state.stanley_druckenmiller_section,
            "benjamin_graham_section": self.state.benjamin_graham_section,
            "charlie_munger_section": self.state.charlie_munger_section,
            "bill_ackman_section": self.state.bill_ackman_section,
            "risk_analysis_section": self.state.risk_analysis_section,
            "future_outlook_section": self.state.future_outlook_section,
            "critic_guidelines": critic_guidelines,
            "editor_guidelines": editor_guidelines,
            "writer_guidelines": writer_guidelines,
        }

        investment_recommendations_section = (
            InvestmentRecommendationCrew().crew().kickoff(inputs=inputs)
        )

        self.state.investment_recommendations_section = (
            investment_recommendations_section.raw
        )
        print("Investment recommendations section generated ✅")
        print(
            f"🪙 Number of tokens used: {investment_recommendations_section.token_usage.total_tokens:,}"
        )
        self.state.total_token_usage += (
            investment_recommendations_section.token_usage.total_tokens
        )

    @timeit
    @listen(generate_investment_recommendations_section)
    def generate_executive_summary(self):
        print("Generating executive summary...")

        report = f"""
        {self.state.company_name}
        {self.state.market_and_industry_section}
        {self.state.competitor_landscape_section}
        {self.state.financial_data_section}
        {self.state.trends_section}
        {self.state.analyst_insights_section}
        {self.state.warren_buffet_section}
        {self.state.cathie_wood_section}
        {self.state.stanley_druckenmiller_section}
        {self.state.benjamin_graham_section}
        {self.state.charlie_munger_section}
        {self.state.bill_ackman_section}
        {self.state.risk_analysis_section}
        {self.state.news_and_research_section}
        {self.state.future_outlook_section}
        {self.state.investment_recommendations_section}
        """

        inputs = {
            "company_name": self.state.company_name,
            "company_ticker": self.state.company_ticker,
            "company_website": self.state.company_website,
            "industry": self.state.industry,
            "date_iso": self.state.date_iso,
            "date_us": self.state.date_us,
            "critic_guidelines": critic_guidelines,
            "editor_guidelines": editor_guidelines,
            "writer_guidelines": writer_guidelines,
            "report": report,
        }

        executive_summary = ExecutiveSummaryCrew().crew().kickoff(inputs=inputs)

        self.state.executive_summary = executive_summary.raw
        print("Executive summary generated ✅")
        print(
            f"🪙 Number of tokens used: {executive_summary.token_usage.total_tokens:,}"
        )
        self.state.total_token_usage += executive_summary.token_usage.total_tokens

    @timeit
    @listen(generate_executive_summary)
    def plot_candlestick_30_days(self):
        print("Plotting candlestick 30 days...")

        try:
            candlestick_30_days_plot = plot_candlestick_chart(
                self.state.company_ticker, PLOTS_DIR, TIMESTAMP
            )
            print(f"📊 OHLC 30 days plot saved to {candlestick_30_days_plot}\n")
        except Exception as e:
            print(f"❌ Error plotting candlestick 30 days: {str(e)}")

        if candlestick_30_days_plot:
            self.state.executive_summary += (
                f"\n\n![30 day candlestick chart]({candlestick_30_days_plot})"
            )
            self.state.executive_summary += f"<figcaption>Candlestick chart showing {self.state.company_ticker} price action over a 30-day period.</figcaption>"
        else:
            print("❌ Candlestick 30 days plot not generated. Skipping...")

    @timeit
    @listen(plot_candlestick_30_days)
    def generate_conclusion(self):
        print("Generating conclusion...")

        report = f"""
        {self.state.executive_summary}
        {self.state.market_and_industry_section}
        {self.state.competitor_landscape_section}
        {self.state.financial_data_section}
        {self.state.trends_section}
        {self.state.analyst_insights_section}
        {self.state.warren_buffet_section}
        {self.state.cathie_wood_section}
        {self.state.stanley_druckenmiller_section}
        {self.state.benjamin_graham_section}
        {self.state.charlie_munger_section}
        {self.state.bill_ackman_section}
        {self.state.risk_analysis_section}
        {self.state.news_and_research_section}
        {self.state.future_outlook_section}
        {self.state.investment_recommendations_section}
        """

        inputs = {
            "company_name": self.state.company_name,
            "date_iso": self.state.date_iso,
            "date_us": self.state.date_us,
            "writer_guidelines": writer_guidelines,
            "report": report,
        }

        conclusion = ConclusionCrew().crew().kickoff(inputs=inputs)

        self.state.conclusion = conclusion.raw

        self.state.conclusion += dedent("""\n\n\n\n\n
        <div class="final-disclaimer">
        <strong>Important:</strong>
        This report is for educational purposes only and does not constitute real investment advice.
        It is not intended to be a recommendation or solicitation to buy or sell any security.
        The information provided is based on publicly available data and should only be used for informational purposes.
        The author is not a licensed financial advisor and is not liable for any investment decisions made based on the information provided in this report.
        <br><br>
        The investment analyses in the style of Warren Buffet, Charlie Munger, Benjamin Graham, Cathie Wood, Stanley Druckenmiller, and Bill Ackman are caricatures of their investment styles and not their real words or opinions.
        The opinions expressed do not necessarily reflect the opinions of the author's employer or any other person or organization.
        </div>
        """)

        print("Conclusion generated ✅")
        print(f"🪙 Number of tokens used: {conclusion.token_usage.total_tokens:,}")
        self.state.total_token_usage += conclusion.token_usage.total_tokens

    @timeit
    @listen(generate_conclusion)
    def save_markdown_report(self) -> Path:
        print("Saving report...")

        intro = dedent(f"""
        # **Equity Research Report on _{self.state.company_name}_ ({self.state.company_ticker})**

        Official website: [{self.state.company_website.lstrip("https://")}]({self.state.company_website})

        Prepared on {datetime.now().strftime("%B %d, %Y")} for {os.getenv("CLIENT")}
        """)

        report_path_md = (
            OUTPUT_DIR
            / f"{self.state.company_ticker}_Stock_Analysis_Report_{TIMESTAMP}.md"
        )

        with open(report_path_md, "w") as f:
            f.write(f"{intro}\n\n")
            f.write(f"{self.state.executive_summary}\n\n")
            f.write(f"{self.state.market_and_industry_section}\n\n")
            f.write(f"{self.state.competitor_landscape_section}\n\n")
            f.write(f"{self.state.financial_data_section}\n\n")
            f.write(f"{self.state.trends_section}\n\n")
            f.write(f"{self.state.analyst_insights_section}\n\n")
            f.write(f"{self.state.warren_buffet_section}\n\n")
            f.write(f"{self.state.cathie_wood_section}\n\n")
            f.write(f"{self.state.stanley_druckenmiller_section}\n\n")
            f.write(f"{self.state.benjamin_graham_section}\n\n")
            f.write(f"{self.state.charlie_munger_section}\n\n")
            f.write(f"{self.state.bill_ackman_section}\n\n")
            f.write(f"{self.state.risk_analysis_section}\n\n")
            f.write(f"{self.state.news_and_research_section}\n\n")
            f.write(f"{self.state.future_outlook_section}\n\n")
            f.write(f"{self.state.investment_recommendations_section}\n\n")
            f.write(f"{self.state.conclusion}\n\n")

        print(f"Report saved to {report_path_md} ✅\n")
        self.state.report_path_md = str(report_path_md)

        return report_path_md

    @timeit
    @listen(save_markdown_report)
    def convert_report_to_pdf(self, report_path_md: Path):
        print("Converting final markdown report to PDF...")
        report_path_pdf = convert_md_to_pdf(report_path_md)
        print(f"Report PDF saved to {report_path_pdf}")
        self.state.report_path_pdf = str(report_path_pdf)
        return report_path_pdf

    @timeit
    @listen(and_(save_markdown_report, convert_report_to_pdf))
    def draft_email_with_attachment(self):
        print("Drafting email...")

        try:
            token_file_path = "src/stock_analyser/tools/token.json"
            with open(token_file_path, "r") as f:
                token_data = json.load(f)
                expiry_date = token_data.get("expiry")
                if expiry_date:
                    expiry_date = datetime.fromisoformat(expiry_date)
                    if expiry_date < datetime.now():
                        print("Token expired. Re-authentication required.")
                        os.remove(token_file_path)

        except FileNotFoundError:
            print("Token file not found. Re-authentication required.")
        except json.JSONDecodeError:
            print("Error decoding token file. Re-authentication required.")
            os.remove(token_file_path)
        except Exception as e:
            print(f"Unexpected error: {str(e)}")

        with open(self.state.report_path_md, "r") as f:
            report_text = f.read()

        inputs = {
            "company_name": self.state.company_name,
            "date_us": self.state.date_us,
            "report_text": report_text,
            "report_pdf": self.state.report_path_pdf,
            "sender": os.getenv("SENDER"),
            "client": os.getenv("CLIENT"),
        }

        gmail_draft = GmailAttachmentCrew().crew().kickoff(inputs=inputs)

        print(f"🪙 Number of tokens used: {gmail_draft.token_usage.total_tokens:,}")
        self.state.total_token_usage += gmail_draft.token_usage.total_tokens
        print("Email draft complete 👍")

    @timeit
    @listen(draft_email_with_attachment)
    def print_total_token_usage(self):
        print(f"Total tokens used: {self.state.total_token_usage:,}")

        completion_time = time.time()
        total_time = completion_time - beginning_time
        minutes, seconds = divmod(total_time, 60)
        print(f"Total time taken: {int(minutes)} minutes and {int(seconds)} seconds")

        with open(
            f"/home/j/ai/crewAI/finance/stock_analyser/timings/timeit_{TIMESTAMP}.txt",
            "a",
        ) as f:
            f.write(
                f"Total time taken: {int(minutes)} minutes and {int(seconds)} seconds\n"
            )
            f.write(f"Total tokens used: {self.state.total_token_usage:,}")
            f.write(
                f"Final report saved to: {str(OUTPUT_DIR)}/{self.state.company_ticker}_Stock_Analysis_Report_{TIMESTAMP}.md\n"
            )


def kickoff():
    raw_input = (
        input(
            "\n\033[1;34mEnter the stock ticker symbol of the company you want to analyze or hit 'ENTER' for a random S&P 500 company:\033[0m "
        )
        .strip()
        .upper()
    )
    report_flow = ReportFlow()
    report_flow.kickoff(inputs={"company_ticker": raw_input})


def plot():
    report_flow = ReportFlow()
    report_flow.plot()


if __name__ == "__main__":
    kickoff()
