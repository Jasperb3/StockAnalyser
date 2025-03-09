# competitor_list_tool.py

from typing import Type, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from nasdaq_data.nasdaq_grabber import nasdaq_grabber

class CompetitorListInput(BaseModel):
    """
    Input schema for CompetitorListTool.
    Provide a list of competitor stock tickers for which you want to
    retrieve last sale, net change, percentage change, and market cap.
    Example: ["AAPL", "MSFT", "GOOG"]
    """
    tickers: List[str] = Field(
        ...,
        description="List of competitor stock tickers."
    )

class CompetitorListTool(BaseTool):
    name: str = "Competitor List Tool"
    description: str = (
        "Retrieves last sale, net change, percentage change, and market cap "
        "for a list of competitor tickers from the nasdaq-data SDK."
    )
    args_schema: Type[BaseModel] = CompetitorListInput

    def _run(self, tickers: List[str]) -> str:
        """
        Run method for the CompetitorListTool.
        Retrieves and returns key stats for each ticker in JSON format.
        """
        ng = nasdaq_grabber()
        # Fetch a sufficiently large subset of stocks; filter by tickers
        df_all = ng.nasdaq_stocks(5000)
        filtered_df = df_all[df_all["symbol"].isin(tickers)]
        # Restrict columns of interest
        output_df = filtered_df[["symbol", "lastsale", "netchange", "pctchange", "marketCap"]]
        return output_df.to_json(orient="records")
