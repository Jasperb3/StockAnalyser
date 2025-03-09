import shutil

from stock_analyser.utils.fmp_get_filings import get_most_recent_filing, save_filings
from stock_analyser.utils.constants import KNOWLEDGE_DIR, FILINGS_DIR

class DownloadFilings:
    def __init__(self, state):
        self.state = state

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