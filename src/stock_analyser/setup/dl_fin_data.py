import yfinance as yf
from nasdaq_data.nasdaq_grabber import nasdaq_grabber

from stock_analyser.utils.constants import KNOWLEDGE_DIR


class DownloadFinancialData:
    def __init__(self, state):
        self.state = state

    def download_financial_data(self):
        print("Downloading financial data...")

        grabber_instance = nasdaq_grabber()

        annual_financial_data = grabber_instance.nasdaq_financals(self.state.company_ticker, 1)
        annual_financial_data_path = f"{str(KNOWLEDGE_DIR)}/{self.state.company_ticker}_annual_financial_data.md"
        with open(annual_financial_data_path, "w") as f:
            f.write(annual_financial_data.to_markdown())

        semiannual_financial_data = grabber_instance.nasdaq_financals(self.state.company_ticker, 2)
        semiannual_financial_data_path = f"{str(KNOWLEDGE_DIR)}/{self.state.company_ticker}_semiannual_financial_data.md"
        with open(semiannual_financial_data_path, "w") as f:
            f.write(semiannual_financial_data.to_markdown())

        company = yf.Ticker(self.state.company_ticker)

        historical_prices = company.history(period="max")
        historical_prices_path = f"{str(KNOWLEDGE_DIR)}/{self.state.company_ticker}_historical_prices.md"
        with open(historical_prices_path, "w") as f:
            f.write(historical_prices.to_markdown())

        company_info = company.info
        balance_sheet = company.balance_sheet.to_markdown() if not company.balance_sheet.empty else "❌ No balance sheet found"
        quarterly_income_stmt = company.quarterly_income_stmt.to_markdown() if not company.quarterly_income_stmt.empty else "❌ No quarterly income statement found"
        calendar = company.calendar if company.calendar else "❌ No calendar found"
        analyst_price_targets = company.analyst_price_targets if company.analyst_price_targets else "❌ No analyst price targets found"

        financial_data_path = f"{str(KNOWLEDGE_DIR)}/{self.state.company_ticker}_financial_data.md"
        with open(financial_data_path, "w") as f:
            f.write(f"# **Company info:**\n\n{company_info}\n\n")
            f.write(f"# **Balance sheet:**\n\n{balance_sheet}\n\n")
            f.write(f"# **Quarterly income statement:**\n\n{quarterly_income_stmt}\n\n")
            f.write(f"# **Calendar:**\n\n{calendar}\n\n")
            f.write(f"# **Analyst price targets:**\n\n{analyst_price_targets}\n\n")

        annual_financial_data_path = f"{str(KNOWLEDGE_DIR)}/{self.state.company_ticker}_annual_financial_data.md"
        with open(annual_financial_data_path, "w") as f:
            f.write(annual_financial_data.to_markdown())

        semiannual_financial_data_path = f"{str(KNOWLEDGE_DIR)}/{self.state.company_ticker}_semiannual_financial_data.md"
        with open(semiannual_financial_data_path, "w") as f:
            f.write(semiannual_financial_data.to_markdown())

        print("Financial data downloaded to the Knowledge directory ✅\n")