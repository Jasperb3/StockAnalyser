from typing import Type, List
from google import genai
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import os

gemini_api_key = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=gemini_api_key, http_options={'api_version': 'v1alpha'})

MODEL = 'gemini-2.0-flash'


class CompetitorNewsSearchToolInput(BaseModel):
    """Input schema for GeminiSearchTool."""

    ticker_list: List[str] = Field(..., description="A list of tickers to search the web for.")


class CompetitorNewsSearchTool(BaseTool):
    name: str = "Competitor News Search Tool"
    description: str = (
        "Use this tool to search the web for information about a company."
    )
    args_schema: Type[BaseModel] = CompetitorNewsSearchToolInput

    def _run(self, ticker_list: List[str]) -> str:
        search_tool = {'google_search': {}}
        chat = client.chats.create(model=MODEL, config={'tools': [search_tool]})

        results = []

        for ticker in ticker_list:
            query = f"What is the latest news on {ticker}? Provide a list of recent articles and for each article provide 1) the article title, 2) the URL of the source article, 3) the publisher, and 4) a summary of the article text."
            r = chat.send_message(query)
            results.append(r.text)

        return results
    

if __name__ == "__main__":
    ticker_list = ["NVDA", "GOOG", "AAPL"]
    tool = CompetitorNewsSearchTool()
    print(tool.run(ticker_list))
