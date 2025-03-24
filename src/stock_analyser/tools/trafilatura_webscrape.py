from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from trafilatura import fetch_url, extract

class TrafilaturaWebscrapeToolInput(BaseModel):
    """Input schema for TrafilaturaWebscrape."""

    url: str = Field(..., description="The URL of the website to scrape. Doesn't work with PDFs. Only one URL at a time.")


class TrafilaturaWebscrapeTool(BaseTool):
    name: str = "Trafilatura Webscrape"
    description: str = (
        "Pass this tool a URL as input and it will scrape the web page to extract its content. Doesn't work with PDFs. Only one URL at a time."
    )
    args_schema: Type[BaseModel] = TrafilaturaWebscrapeToolInput

    def _run(self, url: str) -> str:

        if isinstance(url, list):
            url = url[0]

        downloaded = fetch_url(url)
        extracted = extract(downloaded, output_format="markdown", with_metadata=True)
        if extracted is None:
            return "Could not extract content from the provided URL. Please try a different URL."
        return extracted
