from typing import Type
from google import genai
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import os

gemini_api_key = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=gemini_api_key, http_options={'api_version': 'v1alpha'})

MODEL = 'gemini-3.1-flash-lite'


class CompanyNewsSearchToolInput(BaseModel):
    """Input schema for GeminiSearchTool."""

    ticker: str = Field(..., description="The ticker to search the web for.")


class CompanyNewsSearchTool(BaseTool):
    name: str = "Company News Search Tool"
    description: str = (
        "Use this tool to search the web for information about a company."
    )
    args_schema: Type[BaseModel] = CompanyNewsSearchToolInput

    def _run(self, ticker: str) -> str:
        search_tool = {'google_search': {}}
        chat = client.chats.create(model=MODEL, config={'tools': [search_tool]})

        query = f"What is the latest news on {ticker}?"
        r = chat.send_message(query)

        return r.text
    

if __name__ == "__main__":
    ticker = "NVDA"
    tool = CompanyNewsSearchTool()
    print(tool.run(ticker))
