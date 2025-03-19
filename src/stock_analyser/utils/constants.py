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

REL_KNOW_DIR = f"crewOutputs/{TIMESTAMP}"
REL_KNOW_DIR_ABS = Path(REL_KNOW_DIR).resolve()
REL_KNOW_DIR_ABS.mkdir(parents=True, exist_ok=True)


FILINGS_DIR = Path("/home/j/ai/crewAI/finance/stock_analyser/filings").resolve()
FILINGS_DIR.mkdir(parents=True, exist_ok=True)

DB_DIR = Path("/home/j/ai/crewAI/finance/stock_analyser/db").resolve()

CSS_FILE = Path("/home/j/ai/crewAI/finance/stock_analyser/src/stock_analyser/utils/styling.css").resolve()

BACKOFF_TIME = 20