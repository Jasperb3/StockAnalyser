from datetime import datetime
from pathlib import Path

now = datetime.now()

TIMESTAMP = now.strftime("%Y%m%d_%H%M%S")

OUTPUT_DIR = Path(f"/home/j/ai/crewAI/finance/stock_analyser/final_reports/{now.strftime('%Y-%m-%d')}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

KNOWLEDGE_DIR = Path("/home/j/ai/crewAI/finance/stock_analyser/qdrant_knowledge").resolve()
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

REL_KNOW_DIR = "qdrant_knowledge"

FILINGS_DIR = Path("/home/j/ai/crewAI/finance/stock_analyser/filings").resolve()
FILINGS_DIR.mkdir(parents=True, exist_ok=True)

DB_DIR = Path("/home/j/ai/crewAI/finance/stock_analyser/db").resolve()