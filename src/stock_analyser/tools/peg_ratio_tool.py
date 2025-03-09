# peg_ratio_tool.py

from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from nasdaq_data.nasdaq_grabber import nasdaq_grabber

class PegRatioInput(BaseModel):
    """
    Input schema for PegRatioTool.
    Provide a single stock ticker symbol.
    """
    ticker: str = Field(..., description="Stock ticker symbol, e.g. 'AAPL'.")

class PEGRatioTool(BaseTool):
    name: str = "PEG Ratio Tool"
    description: str = "Retrieves the PEG ratio for a given stock ticker."
    args_schema: Type[BaseModel] = PegRatioInput

    def _run(self, ticker: str) -> str:
        """
        Run method for the PegRatioTool.
        Retrieves PEG ratio data for the ticker.
        Returns JSON-formatted results.
        """
        ng = nasdaq_grabber()
        df = ng.nasdaq_data(ticker, 2)  # 2 => PEG Ratio
        return df.to_json(orient="records")
