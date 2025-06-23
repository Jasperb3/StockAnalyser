import os
import re
import requests
import html2text
import fmpsdk
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
apikey = os.environ.get("FMP_API_KEY")


def get_most_recent_filing(symbol: str, filing_type: str) -> str:
    filings = fmpsdk.sec_filings(apikey=apikey, symbol=symbol, filing_type=filing_type)
    url = filings[0]["finalLink"]

    headers = {
        "User-Agent": "crewai.com bisan@crewai.com",
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    h = html2text.HTML2Text()
    h.ignore_links = False
    text = h.handle(response.content.decode("utf-8"))

    text = re.sub(r"[^a-zA-Z$0-9\s\n]", "", text)
    return text


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
