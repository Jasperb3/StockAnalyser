import os
import time
import random
import yfinance as yf
from pathlib import Path
from textwrap import dedent
from datetime import datetime

from .plotting.plot_risk_severity import plot_risk_severity
from .plotting.plot_awesome_oscillator import plot_awesome_oscillator
from .plotting.plotting_tools import plot_competitor_prices, plot_anaylsts_recommendations, plot_OHLC

from .utils.tickers import tickers
from .utils.setup import Setup
from .utils.models import ReportState
from .utils.decorators import timeit
from .utils.constants import TIMESTAMP, OUTPUT_DIR, PLOTS_DIR, KNOWLEDGE_DIR

from crewai.flow.flow import Flow, listen, start
from .crews.news_and_research_crew.news_and_research_crew import NewsAndResearchCrew
from .crews.market_and_industry_crew.market_and_industry_crew import MarketAndIndustryCrew
from .crews.competitor_crew.competitor_crew import CompetitorCrew
from .crews.financial_data_crew.financial_data_crew import FinancialDataCrew
from .crews.analyst_insights_crew.analyst_insights_crew import AnalystInsightsCrew
from .crews.benjamin_graham_crew.benjamin_graham_crew import BenjaminGrahamCrew 
from .crews.warren_buffet_crew.warren_buffet_crew import WarrenBuffetCrew
from .crews.charlie_munger_crew.charlie_munger_crew import CharlieMungerCrew
from .crews.cathie_wood_crew.cathie_wood_crew import CathieWoodCrew
from .crews.bill_ackman_crew.bill_ackman_crew import BillAckmanCrew
from .crews.risk_analysis_crew.risk_analysis_crew import RiskAnalysisCrew
from .crews.risk_severity_crew.risk_severity_crew import RiskSeverityCrew
from .crews.investment_recommendation_crew.investment_recommendation_crew import InvestmentRecommendationCrew
from .crews.future_outlook_crew.future_outlook_crew import FutureOutlookCrew
from .crews.executive_summary_crew.executive_summary_crew import ExecutiveSummaryCrew
from .crews.conclusion_crew.conclusion_crew import ConclusionCrew
from .crews.gmail_crew.gmail_crew import GmailCrew

beginning_time = time.time()

class ReportFlow(Flow[ReportState]):
    knowledge_source = None
    qdrant_tool = None
    setup_instance = None  # Class variable to store the Setup instance


    @timeit
    @start()
    def confirm_company_stock(self):
        while True:
            self.state.company_ticker = input("\n\033[1;34mEnter the stock ticker symbol of the company you want to analyse or hit 'ENTER' for a random ticker:\033[0m ").strip().upper()
            if self.state.company_ticker == "":
                while True:
                    self.state.company_ticker = random.choice(tickers)
                    history = yf.Ticker(self.state.company_ticker).history(period='7d', interval='1d')
                    if not history.empty:
                        break
                    else:
                        print(f"❌ Invalid ticker: {self.state.company_ticker}. Removing it from the list.")
                        tickers.remove(self.state.company_ticker)
            history = yf.Ticker(self.state.company_ticker).history(period='7d', interval='1d')
            if not history.empty:
                company = yf.Ticker(self.state.company_ticker)

                self.state.company_name = company.info["longName"] if "longName" in company.info else company.info["shortName"] if "shortName" in company.info else company.info["longBusinessSummary"] if "longBusinessSummary" in company.info else "Unknown company name"
                self.state.industry = company.info["industry"] if "industry" in company.info else "Unknown industry"
                self.state.company_website = company.info["website"] if "website" in company.info else "Unknown website"
                
                print(f"Ticker: \033[1;37m{self.state.company_ticker}\033[0m")
                print(f"Company name: \033[1;37m{self.state.company_name}\033[0m")
                print(f"Industry: \033[1;37m{self.state.industry}\033[0m")
                print(f"Website: \033[1;37m{self.state.company_website}\033[0m\n")
                break
            else:
                print("❌ Invalid ticker. Please try again.\n")


    @timeit
    @listen(confirm_company_stock)
    def set_up(self):
        # Create a Setup instance and store it as a class variable
        ReportFlow.setup_instance = Setup(self.state)

        ReportFlow.setup_instance.check_knowledge_dir()
        ReportFlow.setup_instance.download_filings()
        ReportFlow.setup_instance.download_financial_data()
        
        # Set up Qdrant vector search
        qdrant_setup_success = ReportFlow.setup_instance.setup_qdrant_vector_search()
        
        if qdrant_setup_success and ReportFlow.setup_instance.qdrant_tool:
            print("Using Qdrant vector search tool from setup ✅\n")
            # Store the qdrant_tool from setup
            ReportFlow.qdrant_tool = ReportFlow.setup_instance.qdrant_tool
        else:
            print("⚠️ Failed to set up Qdrant vector search.")
            print("⚠️ Please check QDRANT_CLUSTER_URL, QDRANT_API_KEY, and GEMINI_API_KEY environment variables.")
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
            "industry": self.state.industry
        }

        news_and_research_section = (
            NewsAndResearchCrew()
            .crew()
            .kickoff(inputs=inputs)
        )

        self.state.news_and_research_section = news_and_research_section.raw
        print("News and research section generated ✅")
        print(f"🪙 Number of tokens used: {news_and_research_section.token_usage.total_tokens:,}")
        self.state.total_token_usage += news_and_research_section.token_usage.total_tokens
        
        # Process any new markdown files created by this crew
        if ReportFlow.setup_instance:
            ReportFlow.setup_instance.process_new_markdown_files()


    @timeit
    @listen(generate_news_and_research_section)
    def generate_market_and_industry_section(self):
        print("Generating market and industry section...")

        inputs={
            "company_name": self.state.company_name,
            "company_ticker": self.state.company_ticker,
            "company_website": self.state.company_website,
            "industry": self.state.industry,
            "date_iso": self.state.date_iso,
            "date_us": self.state.date_us
        }

        market_and_industry_section = (
            MarketAndIndustryCrew(qdrant_tool=ReportFlow.qdrant_tool)
            .crew()
            .kickoff(inputs=inputs)
        )

        self.state.market_and_industry_section = market_and_industry_section.raw
        print("Market and industry section generated ✅")
        print(f"🪙 Number of tokens used: {market_and_industry_section.token_usage.total_tokens:,}")
        self.state.total_token_usage += market_and_industry_section.token_usage.total_tokens
        
        # Process any new markdown files created by this crew
        if ReportFlow.setup_instance:
            ReportFlow.setup_instance.process_new_markdown_files()


    @timeit
    @listen(generate_market_and_industry_section)
    def generate_competitor_landscape_section(self):
        print("Generating competitor landscape section...")

        inputs = {
            "company_ticker": self.state.company_ticker,
            "company_name": self.state.company_name,
            "market_and_industry_section": self.state.market_and_industry_section,
            "date_iso": self.state.date_iso,
            "date_us": self.state.date_us
        }

        competitor_landscape = (
            CompetitorCrew()
            .crew()
            .kickoff(inputs=inputs)
        )

        self.state.competitor_landscape_section = competitor_landscape.pydantic.content
        print(f"Competitor landscape section generated ✅")
        print(f"🪙 Number of tokens used: {competitor_landscape.token_usage.total_tokens:,}")
        self.state.total_token_usage += competitor_landscape.token_usage.total_tokens

        # Plot competitor prices

        ticker_list = [self.state.company_ticker]

        for competitor in competitor_landscape.pydantic.competitors:
            ticker_list.append(competitor.ticker)

        print(f"Competitors of {self.state.company_ticker}: {ticker_list}")

        print("Plotting competitor prices...")

        try:
            competitor_prices_plot = plot_competitor_prices(ticker_list, PLOTS_DIR, TIMESTAMP)
            print(f"📊 Competitor prices plot saved to {competitor_prices_plot}\n")
        except Exception as e:
            print(f"❌ Error plotting competitor prices: {str(e)}")
            competitor_prices_plot = None

        if competitor_prices_plot:
            self.state.competitor_landscape_section += f"\n\n### Competitor Prices\n![Competitor Price Plot]({competitor_prices_plot})"
        else:
            print("❌ Competitor prices plot not generated. Skipping...")
            
        # Process any new markdown files created by this crew
        if ReportFlow.setup_instance:
            ReportFlow.setup_instance.process_new_markdown_files()


    @timeit
    @listen(generate_competitor_landscape_section)
    def plot_OHLC_30_days(self):
        print("Plotting OHLC 30 days...")

        try:
            ohlc_30_days_plot = plot_OHLC(self.state.company_ticker, PLOTS_DIR, TIMESTAMP)
            print(f"📊 OHLC 30 days plot saved to {ohlc_30_days_plot}\n")
            self.state.competitor_landscape_section += f"\n\n### {self.state.company_name} OHLC for the last 30 days\n![OHLC 30 days Plot]({ohlc_30_days_plot})"
        except Exception as e:
            print(f"❌ Error plotting OHLC 30 days: {str(e)}")


    @timeit
    @listen(plot_OHLC_30_days)
    def generate_financial_data_section(self):
        print("Generating financial data section...")

        inputs={
            "company_name": self.state.company_name,
            "company_ticker": self.state.company_ticker,
            "company_website": self.state.company_website,
            "industry": self.state.industry,
            "date_iso": self.state.date_iso,
            "date_us": self.state.date_us
        }

        financial_data_section = (
            FinancialDataCrew(qdrant_tool=ReportFlow.qdrant_tool)
            .crew()
            .kickoff(inputs=inputs)
        )

        self.state.financial_data_section = financial_data_section.raw
        print("Financial data section generated ✅")
        print(f"🪙 Number of tokens used: {financial_data_section.token_usage.total_tokens:,}")
        self.state.total_token_usage += financial_data_section.token_usage.total_tokens
        
        # Process any new markdown files created by this crew
        if ReportFlow.setup_instance:
            ReportFlow.setup_instance.process_new_markdown_files()


    @timeit
    @listen(generate_financial_data_section)
    def generate_analyst_insights_section(self):
        print("Generating analyst insights section...")

        inputs = {
            "company_name": self.state.company_name,
            "company_ticker": self.state.company_ticker,
            "company_website": self.state.company_website,
            "industry": self.state.industry,
            "date_iso": self.state.date_iso,
            "date_us": self.state.date_us
        }

        analyst_insights_section = (
            AnalystInsightsCrew(qdrant_tool=ReportFlow.qdrant_tool)
            .crew()
            .kickoff(inputs=inputs)
        )

        self.state.analyst_insights_section = analyst_insights_section.raw
        print("Analyst insights section generated ✅")
        print(f"🪙 Number of tokens used: {analyst_insights_section.token_usage.total_tokens:,}")
        self.state.total_token_usage += analyst_insights_section.token_usage.total_tokens
        
        # Process any new markdown files created by this crew
        if ReportFlow.setup_instance:
            ReportFlow.setup_instance.process_new_markdown_files()


    @timeit
    @listen(generate_analyst_insights_section)
    def plot_analyst_recommendations(self):
        print("Plotting analyst recommendations...")

        try:
            analyst_recommendations_plot = plot_anaylsts_recommendations(self.state.company_ticker, PLOTS_DIR, TIMESTAMP)
            print(f"📊 Analyst recommendations plot saved to {analyst_recommendations_plot}\n")
        except Exception as e:
            print(f"❌ Error plotting analyst recommendations: {str(e)}")

        if analyst_recommendations_plot:
            self.state.analyst_insights_section += f"\n![Analyst Recommendations Plot]({analyst_recommendations_plot})"
        else:
            print("❌ Analyst recommendations plot not generated. Skipping...")


    @timeit
    @listen(plot_analyst_recommendations)
    def generate_warren_buffet_section(self):
        print("Generating Warren Buffet section...")

        inputs = {
            "company_name": self.state.company_name,
            "company_ticker": self.state.company_ticker,
            "date_us": self.state.date_us
        }

        warren_buffet_section = (
            WarrenBuffetCrew()
            .crew()
            .kickoff(inputs=inputs)
        )

        self.state.warren_buffet_section = warren_buffet_section.raw
        print("Warren Buffet section generated ✅")
        print(f"🪙 Number of tokens used: {warren_buffet_section.token_usage.total_tokens:,}")
        self.state.total_token_usage += warren_buffet_section.token_usage.total_tokens
        
        # Process any new markdown files created by this crew
        if ReportFlow.setup_instance:
            ReportFlow.setup_instance.process_new_markdown_files()


    @timeit
    @listen(generate_warren_buffet_section)
    def generate_cathie_wood_section(self):
        print("Generating Cathie Wood section...")

        inputs = {
            "company_name": self.state.company_name,
            "company_ticker": self.state.company_ticker,
            "date_us": self.state.date_us
        }

        cathie_wood_section = (
            CathieWoodCrew()
            .crew()
            .kickoff(inputs=inputs)
        )

        self.state.cathie_wood_section = cathie_wood_section.raw
        print("Cathie Wood section generated ✅")
        print(f"🪙 Number of tokens used: {cathie_wood_section.token_usage.total_tokens:,}")
        self.state.total_token_usage += cathie_wood_section.token_usage.total_tokens
        
        # Process any new markdown files created by this crew
        if ReportFlow.setup_instance:
            ReportFlow.setup_instance.process_new_markdown_files()

        time.sleep(30)
        for remaining in range(30, 0, -1):
            print(f"⏳ Pausing for {remaining} seconds to avoid rate limit...", end="\r")
            time.sleep(1)
        print("Resuming...")
    

    @timeit
    @listen(generate_cathie_wood_section)
    def generate_benjamin_graham_section(self):
        print("Generating Benjamin Graham section...")

        inputs = {
            "company_name": self.state.company_name,
            "company_ticker": self.state.company_ticker,
            "date_us": self.state.date_us
        }

        benjamin_graham_section = (
            BenjaminGrahamCrew()
            .crew()
            .kickoff(inputs=inputs)
        )

        self.state.benjamin_graham_section = benjamin_graham_section.raw
        print("Benjamin Graham section generated ✅")
        print(f"🪙 Number of tokens used: {benjamin_graham_section.token_usage.total_tokens:,}")
        self.state.total_token_usage += benjamin_graham_section.token_usage.total_tokens
        
        # Process any new markdown files created by this crew
        if ReportFlow.setup_instance:
            ReportFlow.setup_instance.process_new_markdown_files()

        time.sleep(30)
        for remaining in range(30, 0, -1):
            print(f"⏳ Pausing for {remaining} seconds to avoid rate limit...", end="\r")
            time.sleep(1)
        print("Resuming...")


    @timeit
    @listen(generate_benjamin_graham_section)
    def generate_charlie_munger_section(self):
        print("Generating Charlie Munger section...")

        inputs = {
            "company_name": self.state.company_name,
            "company_ticker": self.state.company_ticker,
            "date_us": self.state.date_us
        }

        charlie_munger_section = (
            CharlieMungerCrew()
            .crew()
            .kickoff(inputs=inputs)
        )

        self.state.charlie_munger_section = charlie_munger_section.raw
        print("Charlie Munger section generated ✅")
        print(f"🪙 Number of tokens used: {charlie_munger_section.token_usage.total_tokens:,}")
        self.state.total_token_usage += charlie_munger_section.token_usage.total_tokens
        
        # Process any new markdown files created by this crew
        if ReportFlow.setup_instance:
            ReportFlow.setup_instance.process_new_markdown_files()

        time.sleep(30)
        for remaining in range(30, 0, -1):
            print(f"⏳ Pausing for {remaining} seconds to avoid rate limit...", end="\r")
            time.sleep(1)
        print("Resuming...")


    @timeit
    @listen(generate_charlie_munger_section)
    def generate_bill_ackman_section(self):
        print("Generating Bill Ackman section...")

        inputs = {
            "company_name": self.state.company_name,
            "company_ticker": self.state.company_ticker,
            "date_us": self.state.date_us
        }

        bill_ackman_section = (
            BillAckmanCrew()
            .crew()
            .kickoff(inputs=inputs)
        )

        self.state.bill_ackman_section = bill_ackman_section.raw
        print("Bill Ackman section generated ✅")
        print(f"🪙 Number of tokens used: {bill_ackman_section.token_usage.total_tokens:,}")
        self.state.total_token_usage += bill_ackman_section.token_usage.total_tokens
        
        # Process any new markdown files created by this crew
        if ReportFlow.setup_instance:
            ReportFlow.setup_instance.process_new_markdown_files()


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
            "date_us": self.state.date_us
        }

        risk_analysis_section = (
            RiskAnalysisCrew(qdrant_tool=ReportFlow.qdrant_tool)
            .crew()
            .kickoff(inputs=inputs)
        )

        self.state.risk_analysis_section = risk_analysis_section.raw
        print("Risk analysis section generated ✅")
        print(f"🪙 Number of tokens used: {risk_analysis_section.token_usage.total_tokens:,}")
        self.state.total_token_usage += risk_analysis_section.token_usage.total_tokens
        
        # Process any new markdown files created by this crew
        if ReportFlow.setup_instance:
            ReportFlow.setup_instance.process_new_markdown_files()


    @timeit
    @listen(generate_risk_analysis_section)
    def plot_risk_severity(self):
        print("Plotting risk severity...")

        inputs = {
            "company_name": self.state.company_name,
            "risk_analysis": self.state.risk_analysis_section,
            "date_iso": self.state.date_iso
        }

        identified_risks = (
            RiskSeverityCrew()
            .crew()
            .kickoff(inputs=inputs).pydantic
        )

        risks = identified_risks.risk_severity_list

        try:
            risk_severity_plot = plot_risk_severity(risks, PLOTS_DIR, TIMESTAMP)
            print(f"Risk severity plot saved to {risk_severity_plot}\n")

            self.state.risk_analysis_section += f"\n![Risk Severity Plot]({risk_severity_plot})"
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
            "date_us": self.state.date_us
        }

        future_outlook_section = (
            FutureOutlookCrew(qdrant_tool=ReportFlow.qdrant_tool)
            .crew()
            .kickoff(inputs=inputs)
        )

        self.state.future_outlook_section = future_outlook_section.raw
        print("Future outlook section generated ✅")
        print(f"🪙 Number of tokens used: {future_outlook_section.token_usage.total_tokens:,}")
        self.state.total_token_usage += future_outlook_section.token_usage.total_tokens
        
        # Process any new markdown files created by this crew
        if ReportFlow.setup_instance:
            ReportFlow.setup_instance.process_new_markdown_files()


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
            "analyst_insights_section": self.state.analyst_insights_section,
            "warren_buffet_section": self.state.warren_buffet_section,
            "cathie_wood_section": self.state.cathie_wood_section,
            "benjamin_graham_section": self.state.benjamin_graham_section,
            "charlie_munger_section": self.state.charlie_munger_section,
            "bill_ackman_section": self.state.bill_ackman_section,
            "risk_analysis_section": self.state.risk_analysis_section,
            "future_outlook_section": self.state.future_outlook_section
        }   

        investment_recommendations_section = (
            InvestmentRecommendationCrew()
            .crew()
            .kickoff(inputs=inputs)
        )

        self.state.investment_recommendations_section = investment_recommendations_section.raw
        print("Investment recommendations section generated ✅")
        print(f"🪙 Number of tokens used: {investment_recommendations_section.token_usage.total_tokens:,}")
        self.state.total_token_usage += investment_recommendations_section.token_usage.total_tokens
        
        # Process any new markdown files created by this crew
        if ReportFlow.setup_instance:
            ReportFlow.setup_instance.process_new_markdown_files()


    @timeit
    @listen(generate_investment_recommendations_section)
    def generate_executive_summary(self):
        print("Generating executive summary...")

        report = f"""
        {self.state.company_name}
        {self.state.market_and_industry_section}
        {self.state.competitor_landscape_section}
        {self.state.financial_data_section}
        {self.state.analyst_insights_section}
        {self.state.warren_buffet_section}
        {self.state.cathie_wood_section}
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
            "report": report
        }   

        executive_summary = (
            ExecutiveSummaryCrew()
            .crew()
            .kickoff(inputs=inputs)
        )

        self.state.executive_summary = executive_summary.raw
        print("Executive summary generated ✅")
        print(f"🪙 Number of tokens used: {executive_summary.token_usage.total_tokens:,}")
        self.state.total_token_usage += executive_summary.token_usage.total_tokens
        
        # Process any new markdown files created by this crew
        if ReportFlow.setup_instance:
            ReportFlow.setup_instance.process_new_markdown_files()


    @timeit
    @listen(generate_executive_summary)
    def plot_historical_stock_data(self):
        print("Plotting historical stock data...")   

        try:
            historical_stock_data_plot = plot_awesome_oscillator(self.state.company_ticker, "1y", PLOTS_DIR, TIMESTAMP)
            print(f"📊 Historical stock data plot saved to {historical_stock_data_plot}\n")
        except Exception as e:
            print(f"❌ Error plotting historical stock data: {str(e)}")

        if historical_stock_data_plot:
            self.state.executive_summary += f"\n\n### Historical Stock Data\n![Historical Stock Data Plot]({historical_stock_data_plot})"
        else:
            print("❌ Historical stock data plot not generated. Skipping...")


    @timeit
    @listen(plot_historical_stock_data)
    def generate_conclusion(self):
        print("Generating conclusion...")

        report = f"""
        {self.state.executive_summary}
        {self.state.market_and_industry_section}
        {self.state.competitor_landscape_section}
        {self.state.financial_data_section}
        {self.state.analyst_insights_section}
        {self.state.warren_buffet_section}
        {self.state.cathie_wood_section}
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
            "report": report
        }

        conclusion = (
            ConclusionCrew()
            .crew()
            .kickoff(inputs=inputs)
        )

        self.state.conclusion = conclusion.raw
        print("Conclusion generated ✅")
        print(f"🪙 Number of tokens used: {conclusion.token_usage.total_tokens:,}")
        self.state.total_token_usage += conclusion.token_usage.total_tokens
        
        # Process any new markdown files created by this crew
        if ReportFlow.setup_instance:
            ReportFlow.setup_instance.process_new_markdown_files()


    @timeit
    @listen(generate_conclusion)
    def save_report(self):
        print("Saving report...")

        intro = dedent(f"""
        # **Research report on _{self.state.company_name}_ ({self.state.company_ticker})**

        Official website: [{self.state.company_website.lstrip("https://")}]({self.state.company_website})

        Prepared on {datetime.now().strftime("%B %d, %Y")} for {os.getenv("CLIENT")}
        """)

        report_path = OUTPUT_DIR / f"{self.state.company_ticker}_Stock_Analysis_Report_{TIMESTAMP}.md"

        with open(report_path, "w") as f:
            f.write(f"{intro}\n\n")
            f.write(f"{self.state.executive_summary}\n\n")
            f.write(f"{self.state.market_and_industry_section}\n\n")
            f.write(f"{self.state.competitor_landscape_section}\n\n")
            f.write(f"{self.state.financial_data_section}\n\n")
            f.write(f"{self.state.analyst_insights_section}\n\n")
            f.write(f"{self.state.warren_buffet_section}\n\n")
            f.write(f"{self.state.cathie_wood_section}\n\n")
            f.write(f"{self.state.benjamin_graham_section}\n\n")
            f.write(f"{self.state.charlie_munger_section}\n\n")
            f.write(f"{self.state.bill_ackman_section}\n\n")
            f.write(f"{self.state.risk_analysis_section}\n\n")
            f.write(f"{self.state.news_and_research_section}\n\n")
            f.write(f"{self.state.future_outlook_section}\n\n")
            f.write(f"{self.state.investment_recommendations_section}\n\n")
            f.write(f"{self.state.conclusion}\n\n")

        print(f"Report saved to {report_path} ✅\n")
        
        # Final processing of any new markdown files
        if ReportFlow.setup_instance:
            ReportFlow.setup_instance.process_new_markdown_files()
        
        return report_path


    @timeit
    @listen(save_report)
    def draft_email(self, report_path: Path):
        print("Drafting email...")

        with open(report_path, "r") as f:
            report = f.read()

        inputs = {
            "company_name": self.state.company_name,
            "date_us": self.state.date_us,
            "report": report,
            "sender": os.getenv("SENDER"),
            "client": os.getenv("CLIENT")
        }

        gmail_draft = GmailCrew().crew().kickoff(inputs=inputs)

        print(f"🪙 Number of tokens used: {gmail_draft.token_usage.total_tokens:,}")
        self.state.total_token_usage += gmail_draft.token_usage.total_tokens
        print("Email draft complete 👍")


    @timeit
    @listen(draft_email)
    def print_total_token_usage(self):
        print(f"Total tokens used: {self.state.total_token_usage:,}")

        completion_time = time.time()
        total_time = completion_time - beginning_time
        minutes, seconds = divmod(total_time, 60)
        print(f"Total time taken: {int(minutes)} minutes and {int(seconds)} seconds")

        with open(f"/home/j/ai/crewAI/finance/stock_analyser/timings/timeit_{TIMESTAMP}.txt", "a") as f:
            f.write(f"Total time taken: {int(minutes)} minutes and {int(seconds)} seconds\n")
            f.write(f"Total tokens used: {self.state.total_token_usage:,}")
            f.write(f"Final report saved to: {str(OUTPUT_DIR)}/{self.state.company_ticker}_Stock_Analysis_Report_{TIMESTAMP}.md")
            f.write("-" * 50)


def kickoff():
    report_flow = ReportFlow()
    report_flow.kickoff()


def plot():
    report_flow = ReportFlow()
    report_flow.plot()


if __name__ == "__main__":
    kickoff()
