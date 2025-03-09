import os
import requests
from typing import Type, List, Dict, Any
from urllib.parse import quote_plus
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class GoogleSearchToolInput(BaseModel):
    """Input schema for GoogleSearchTool."""
    query: str = Field(..., description="The search query to look up on Google.")
    num_results: int = Field(default=10, description="Number of search results to return (default: 10)")

class GoogleSearchTool(BaseTool):
    name: str = "GoogleSearchTool"
    description: str = "Performs a Google search using Custom Search API and returns relevant results including titles, descriptions, and URLs."
    args_schema: Type[BaseModel] = GoogleSearchToolInput
    api_key: str
    cx: str
    
    def __init__(self, api_key: str, cx: str):
        super().__init__(api_key=api_key, cx=cx)
        self.api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        self.cx = os.getenv("SEARCH_ENGINE_ID")

    def _run(self, query: str, num_results: int = 10) -> str:
        try:
            search_results = self._perform_search(query, num_results)
            formatted_results = "\n\n".join([
                f"Title: {result['title']}\nURL: {result['link']}\nDescription: {result.get('snippet', '')}"
                for result in search_results
            ])
            return formatted_results
        except Exception as e:
            return f"Error performing search: {str(e)}"

    def _perform_search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        url = f"https://www.googleapis.com/customsearch/v1?key={self.api_key}&cx={self.cx}&q={quote_plus(query)}&num={min(num_results, 10)}"
        
        response = requests.get(url)
        response.raise_for_status()
        results = response.json()
        
        if "items" not in results:
            return []
            
        return results["items"][:num_results]

if __name__ == "__main__":
    # Replace these with your actual API credentials
    api_key = os.getenv("GOOGLE_API_KEY")
    cx = os.getenv("SEARCH_ENGINE_ID")
    
    # Create an instance of the search tool
    search_tool = GoogleSearchTool(api_key=api_key, cx=cx)
    
    # Test search query
    test_query = "latest premier league results"
    
    # Execute the search
    results = search_tool._run(
        query=test_query,
        num_results=5
    )
    
    # Print the results
    print(f"Search Results for: {test_query}\n")
    print(results) 