from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import yfinance as yf
from groq import Groq
from trafilatura import fetch_url, extract
from datetime import datetime
from typing import Type

# Define the input schema using Pydantic
class YFinanceNewsToolInput(BaseModel):
    """Input schema for YFinanceNewsTool."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)")

# Define the tool class
class YFinanceNewsTool(BaseTool):
    name: str = "YFinance News Tool"
    description: str = "Fetches and analyzes news for a given ticker using yfinance."
    args_schema: Type[BaseModel] = YFinanceNewsToolInput

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

    def _run(self, ticker: str) -> str:
        """
        Fetches news and research for a given ticker using yfinance.
        :param ticker: Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
        :return: A string containing the stock's news.
        """
        # get list of news
        news = yf.Search(ticker, news_count=10).news

        # get list of related research
        research = yf.Search(ticker, include_research=True).research

        results = []
        
        for article in news:
            url = article.get('link')
            text = self.get_article_text(url)
            if text:
                results.append({
                    'type': 'news',
                    'title': article.get('title'),
                    'publisher': article.get('publisher'),
                    'date_published': datetime.fromtimestamp(article.get('providerPublishTime')).strftime('%B %d, %Y'),
                    'url': url,
                    'summary': text
                })
        
        for article in research:
            url = article.get('link')
            text = self.get_article_text(url)
            if text:
                results.append({
                    'type': 'research',
                    'title': article.get('title'),
                    'publisher': article.get('publisher'),
                    'date_published': datetime.fromtimestamp(article.get('providerPublishTime')).strftime('%B %d, %Y'),
                    'url': article.get('link'),
                    'summary': text
                })
        
        return results
    

if __name__ == "__main__":
    tool = YFinanceNewsTool()
    print(tool.run("AAPL"))
