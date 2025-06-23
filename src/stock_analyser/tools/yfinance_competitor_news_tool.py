from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import yfinance as yf
from trafilatura import fetch_url, extract
from typing import Type, List
from pprint import pprint
from groq import Groq
from datetime import datetime


# Define the input schema using Pydantic
class YFinanceCompetitorNewsToolInput(BaseModel):
    """Input schema for YFinanceCompetitorNewsTool."""

    tickers: list[str] = Field(
        ...,
        description="List of competitor ticker symbols (e.g., ['AAPL', 'NVDA', 'GOOG'] for Apple Inc., NVIDIA Corp., and Alphabet Inc.)",
    )


# Define the tool class
class YFinanceCompetitorNewsTool(BaseTool):
    name: str = "YFinance Competitor News Tool"
    description: str = "Fetches and analyzes news for a given ticker using yfinance."
    args_schema: Type[BaseModel] = YFinanceCompetitorNewsToolInput

    def get_article_text(self, url):
        try:
            downloaded = fetch_url(url)
            return extract(downloaded)
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return ""

    def summarise_article(self, text):
        try:
            client = Groq()

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analyst assistant specialising in extracting key company-specific insights from financial news articles. Your task is to produce concise, information-dense summaries that include only the most relevant and material information about the company that is the subject of the article. You must ignore unrelated context, general market commentary, and filler content. You write in clear, objective, plain English with no formatting, lists, or bullet points.",
                    },
                    {
                        "role": "user",
                        "content": f"Given the full text of a financial article, write a concise summary that includes only the most important information relevant to the company that is the main subject of the article. The summary must: Focus exclusively on the company’s performance, strategy, financials, outlook, risks, leadership, operations, deals, and any regulatory or macroeconomic factors directly affecting it. Omit general market context, irrelevant background, commentary on the broader industry unless directly linked to the company’s situation. Maintain the original meaning and prioritise factual accuracy. Use plain language and write in full, coherent sentences grouped into one or more short paragraphs. Avoid bullet points, markdown, headlines, or repetition. Your output should be as short as possible while capturing all material points. Do not include your reasoning or any extraneous explanation. Article: {text}",
                    },
                ],
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=0.5,
                max_completion_tokens=4096,
                top_p=1,
                stop=None,
                stream=False,
            )
            summary = chat_completion.choices[0].message.content

            return summary

        except Exception as e:
            print(f"Error summarising article: {e}")
            return ""

    def _run(self, tickers: list[str]) -> str:
        """
        Fetches news and research for given tickers using yfinance.
        :param tickers: List of competitor ticker symbols (e.g., ['AAPL', 'NVDA', 'GOOG'] for Apple Inc., NVIDIA Corp., and Alphabet Inc.)
        :return: A string containing the stock's news.
        """
        competitor_news = {}
        results = {ticker: [] for ticker in tickers}
        for ticker in tickers:
            # get list of news
            news = yf.Search(ticker, news_count=8).news

            # get list of related research
            research = yf.Search(ticker, include_research=True).research

            competitor_news[ticker] = {"news": news, "research": research}

            # get text of news and research
            for article in news:
                url = article.get("link")
                text = self.get_article_text(url)
                if text:
                    summary = self.summarise_article(text)
                    results[ticker].append(
                        {
                            "type": "news",
                            "title": article.get("title"),
                            "publisher": article.get("publisher"),
                            "date_published": datetime.fromtimestamp(
                                article.get("providerPublishTime")
                            ).strftime("%B %d, %Y"),
                            "url": url,
                            "summary": summary,
                        }
                    )
            # for article in research:
            #     url = article.get("link")
            #     text = self.get_article_text(url)
            #     if text:
            #         summary = self.summarise_article(text)
            #         results[ticker].append(
            #             {
            #                 "type": "research",
            #                 "title": article.get("title"),
            #                 "publisher": article.get("publisher"),
            #                 "date_published": datetime.fromtimestamp(
            #                     article.get("providerPublishTime")
            #                 ).strftime("%B %d, %Y"),
            #                 "url": url,
            #                 "summary": summary,
            #             }
            #         )

        return results


if __name__ == "__main__":
    tool_instance = YFinanceCompetitorNewsTool()
    competitor_news = tool_instance.run(tickers=["NVDA", "AAPL", "GOOG"])
    pprint(competitor_news)
