import os
import re
import requests
import html2text
from pathlib import Path
from dotenv import load_dotenv
from sec_api import QueryApi

load_dotenv()
sec_api_key = os.environ.get("SEC_API_KEY")


def get_most_recent_filing(symbol: str, filing_type: str) -> str:
    """
    Get the most recent SEC filing for a given symbol and filing type.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
        filing_type: Type of filing (e.g., '10-K', '10-Q', '8-K', '6-K')

    Returns:
        Text content of the filing

    Raises:
        ValueError: If no filings are found or API key is missing
        requests.HTTPError: If the filing download fails
    """
    if not sec_api_key:
        raise ValueError("SEC_API_KEY not found in environment variables")

    try:
        # Use sec-api to query for filings
        queryApi = QueryApi(api_key=sec_api_key)

        query = {
            "query": f'ticker:{symbol} AND formType:"{filing_type}"',
            "from": "0",
            "size": "1",
            "sort": [{"filedAt": {"order": "desc"}}]
        }

        filings = queryApi.get_filings(query)

        if not filings or "filings" not in filings or len(filings["filings"]) == 0:
            raise ValueError(f"No {filing_type} filings found for {symbol}")

        # Get the filing URL
        filing = filings["filings"][0]
        url = filing.get("linkToFilingDetails") or filing.get("linkToTxt")

        if not url:
            raise ValueError(f"No valid URL found in filing data for {symbol}")

        # Download the filing
        headers = {
            "User-Agent": "crewai.com bisan@crewai.com",
            "Accept-Encoding": "gzip, deflate",
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Convert HTML to text
        h = html2text.HTML2Text()
        h.ignore_links = False
        text = h.handle(response.content.decode("utf-8"))

        # Clean up the text - remove special characters but keep alphanumeric, $, spaces, and newlines
        text = re.sub(r"[^a-zA-Z$0-9\s\n]", "", text)

        return text

    except Exception as e:
        raise Exception(f"Error fetching {filing_type} filing for {symbol}: {str(e)}")


def save_filings(symbol: str, filing_type: str, filename: Path):
    text = get_most_recent_filing(symbol, filing_type)
    with open(filename, "w") as f:
        f.write(text)


if __name__ == "__main__":
    symbol = input("Enter a symbol: ").upper()
    filing_types = ["10-Q", "10-K", "6-K", "8-K"]

    output_dir = Path(__file__).parent.parent.parent / "filings"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filing_type in filing_types:
        try:
            filings = get_most_recent_filing(symbol, filing_type)
            if filings:
                print(f"Downloading {filing_type} filing for {symbol}...")
            save_filings(
                symbol, filing_type, output_dir / f"{symbol}_{filing_type}.txt"
            )
        except Exception as e:
            print(f"Error downloading {filing_type} filing for {symbol}: {e}")
