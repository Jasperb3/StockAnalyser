import os
import yfinance as yf
from typing import List, Dict
from pydantic import BaseModel
from trafilatura import fetch_url, extract
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

NUMBER_OF_NEWS_ARTICLES = 2

class NewsItem(BaseModel):
    title: str
    sentiment: float

class NewsItemList(BaseModel):
    news_items: List[NewsItem]


def scrape_text(url: str) -> str:
    """
    Scrape the text from a given URL.
    """
    try:
        return extract(fetch_url(url), output_format="markdown", with_metadata=True, fast=True)
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""


def get_news(ticker: str, number_of_articles: int = NUMBER_OF_NEWS_ARTICLES) -> List[Dict[str, str]]:
    """
    Get the latest news for a given ticker.
    """
    company_ticker = yf.Ticker(ticker)

    raw_news_results = company_ticker.news

    news_results = []

    cutoff = number_of_articles

    for news in raw_news_results[:cutoff]:
        try:
            news_results.append({
                "title": news["content"]["title"],
                "source": news["content"]["provider"]["displayName"],
                "date": news["content"]["pubDate"],
                "content": scrape_text(news["content"]["canonicalUrl"]["url"])
            })

        except Exception as e:
            print(f"Error scraping text for {news.get('title', 'No Title')}: {e}")
            cutoff += 1
            if len(news_results) > 0: # Check if news_results is not empty before popping
              news_results.pop()


    # print(f"news_results: {news_results}")
    return news_results


def get_llm_sentiment_scores(news: List[Dict[str, str]], ticker: str) -> List[float]:
    """
    Get the sentiment score for a given news item using an LLM.
    """

    system_prompt = f"""
    You are an expert sentiment analyst tasked with evaluating the emotional tone of news articles. Your analysis will contribute to determining the overall public perception of a specific company.

    Task:
    You will be provided with a list of news articles. For each article, you must assign a sentiment score on a scale of -100 to 100, where:
    -100 represents extremely negative sentiment.
    0 represents neutral sentiment.
    100 represents extremely positive sentiment.

    Evaluation Criteria:
    Your scoring should be based on a comprehensive assessment of the article, considering:

    - The explicit tone and sentiment expressed in the language used (positive, negative, or neutral).
    - The implied sentiment and potential market reaction conveyed through word choice, phrasing, and context.
    - The overall subject matter and its relevance to the company's business, financials, and industry.
    - The presence of heavily loaded words, phrases, or events that could significantly influence investor perception.
    - The overall implications of the article for the company's future prospects, stock value, and market position.
    - Whether the news is likely to cause any short-term fluctuations or a long-term shift in the stock's value.

    Output Format:
    A list of NewsItem objects, each containing the title and score of the article. The output MUST be valid JSON.

    Example Output:
    [
        NewsItem({{
            "title": "Article Title 1",
            "sentiment": 75
        }}),
        NewsItem({{
            "title": "Article Title 2",
            "sentiment": -50
        }}),
        NewsItem({{
            "title": "Article Title 3",
            "sentiment": 0
        }})
    ]
    """

    formatted_news = ""
    for article in news:
        formatted_news += f"Title: {article['title']}\\nContent: {article['content']}\\n\\n"


    user_prompt = f"""
    Please analyze the following news articles in relation to the company {ticker}:

    {formatted_news}
    """
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=NewsItemList
        )


        response = completion.choices[0].message.parsed

        print(f"Response: {response}")
        print(f"Total tokens: {completion.usage.total_tokens}")

        return [item.sentiment for item in response.news_items]
    except Exception as e:
        print(f"Error calling OpenAI: {e}")
        return []



def get_news_sentiment_scores(ticker: str, number_of_articles: int = NUMBER_OF_NEWS_ARTICLES) -> float:
    """
    Get the sentiment score for a given ticker.
    """
    news = get_news(ticker, number_of_articles=number_of_articles)
    if not news:
        return 0.0
    
    llm_analysis = get_llm_sentiment_scores(news, ticker)
    if not llm_analysis:
        return 0.0

    total_score = sum(llm_analysis)

    sentiment_score = round(total_score / len(llm_analysis), 3)

    print(f"Overall sentiment score: {sentiment_score} from {len(news)} articles")
    return sentiment_score


if __name__ == "__main__":
    print(get_news_sentiment_scores("AAPL"))


