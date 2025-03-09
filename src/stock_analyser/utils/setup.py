import shutil
import yfinance as yf
from nasdaq_data.nasdaq_grabber import nasdaq_grabber
from stock_analyser.utils.fmp_get_filings import get_most_recent_filing, save_filings
from stock_analyser.utils.constants import KNOWLEDGE_DIR, FILINGS_DIR, DB_DIR

class Setup:
    def __init__(self, state):
        self.state = state

    
    def check_knowledge_dir(self):
        print("Checking knowledge directory...")
        # Get the expected marker file name
        marker_file = KNOWLEDGE_DIR / f"{self.state.company_ticker}.txt"

        # Archive files if the marker file doesn't exist or if it's a different ticker
        for file in KNOWLEDGE_DIR.glob("*.txt"):
            if file != marker_file:
                archive_dir = KNOWLEDGE_DIR / f"{file.stem}_archive"
                print(f"Archiving existing files to {archive_dir}...")
                archive_dir.mkdir(parents=True, exist_ok=True)
                # Archive .md files
                for old_file in KNOWLEDGE_DIR.glob("*.md"):
                    new_file_path = archive_dir / old_file.name
                    old_file.rename(new_file_path)
                file.unlink()  # Remove the old marker file

                # Remove the db directory
                db = DB_DIR
                if db.exists():
                    shutil.rmtree(db)

        # Create/overwrite the marker file for the current ticker
        marker_file.touch()

        print("Knowledge directory sorted ✅\n")


    def download_filings(self):
        download_required = False
        existing_files = list(FILINGS_DIR.iterdir())

        if existing_files:
            if not any(file.name.startswith(self.state.company_ticker) for file in existing_files):
                for file in existing_files:
                    file.unlink()
                download_required = True
        else:
            download_required = True
            print(f"No existing files found for {self.state.company_name} ({self.state.company_ticker})\n")

        filing_types = ["10-Q", "10-K", "6-K", "8-K"]
        success = True

        if download_required:
            print(f"Downloading SEC filings for {self.state.company_name} ({self.state.company_ticker})...")

            for filing_type in filing_types:
                try:
                    file_path = FILINGS_DIR / f"{self.state.company_ticker}_{filing_type}.txt"
                    try:
                        filings = get_most_recent_filing(self.state.company_ticker, filing_type)
                    except Exception as e:
                        print(f"❌ Error getting {filing_type} filing: {e}")
                        success = False
                        continue

                    if not filings:  # Check for empty content *before* saving
                        print(f"No {filing_type} filings found or empty for {self.state.company_ticker}")
                        success = False
                        continue

                    try:
                        save_filings(self.state.company_ticker, filing_type, file_path)
                        print(f"{filing_type} filing saved to {file_path} ✅")
                        success = True
                    except Exception as e:
                        print(f"❌ Error saving {filing_type} filing: {e}")
                        success = False
                        continue

                    if not file_path.stat().st_size:
                        print(f"❌ No content downloaded for {filing_type}")
                        success = False
                        continue  # Continue to the next filing type

                    # Conditional copy: only if it doesn't exist
                    knowledge_file_path = KNOWLEDGE_DIR / f"{self.state.company_ticker}_{filing_type}.md"
                    if not knowledge_file_path.exists():
                        try:
                            shutil.copyfile(str(file_path), str(knowledge_file_path))
                        except Exception as e:
                            print(f"❌ Error copying {filing_type} to knowledge dir: {e}")
                            success = False # Still track the failure
                            continue

                except Exception as e:  # Catch-all for unexpected errors in the loop
                    print(f"❌ Unexpected error processing {filing_type}: {e}")
                    success = False
                    continue

            if not success:
                print(f"❌ Failed to download required SEC filings for {self.state.company_name} ({self.state.company_ticker}).\nPlease verify the stock ticker and try again.")
            
            print("SEC filings downloaded successfully ✅\n")

        else:
            print(f"Using existing SEC filings for {self.state.company_name} ({self.state.company_ticker})")
            for filing_type in filing_types:
                file_path = FILINGS_DIR / f"{self.state.company_ticker}_{filing_type}.txt"
                knowledge_file_path = KNOWLEDGE_DIR / f"{self.state.company_ticker}_{filing_type}.md"
                if file_path.exists() and not knowledge_file_path.exists():
                    try:
                        shutil.copyfile(str(file_path), str(knowledge_file_path))
                    except Exception as e:
                        print(f"❌ Error copying existing {filing_type} to knowledge dir: {e}")

            print("SEC filings copied to the Knowledge directory ✅\n")

    
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