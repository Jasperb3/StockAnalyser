import os
from stock_analyser.utils.models import CompetitorTickerList
from stock_analyser.utils.agent_llms import RESEARCH_MODEL
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from stock_analyser.tools.gemini_search_tool import GeminiSearchTool
from stock_analyser.tools.trafilatura_webscrape import TrafilaturaWebscrapeTool
from stock_analyser.tools.tavily_search import TavilySearchTool
from stock_analyser.tools.google_search_tool import GoogleSearchTool
from stock_analyser.tools.linkup_search_tool import LinkUpSearchTool
from stock_analyser.tools.yfinance_industry_leaders_tool import YFinanceIndustryLeadersTool
from dotenv import load_dotenv

load_dotenv()

google_search_tool = GoogleSearchTool(api_key=os.getenv("GOOGLE_SEARCH_API_KEY"), cx=os.getenv("SEARCH_ENGINE_ID"))

	
@CrewBase
class CompetitionIdentifierCrew():
	"""CompetitionIdentifierCrew crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# ----Agents----#
	@agent
	def competitor_identifier_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['competitor_identifier_agent'],
			verbose=True,
			tools=[
				GeminiSearchTool(),
				YFinanceIndustryLeadersTool(),
				TavilySearchTool(),
				google_search_tool,
				TrafilaturaWebscrapeTool(),
				LinkUpSearchTool()
			],
			llm=RESEARCH_MODEL,
			output_json=True
		)

	# ----Tasks----#
	@task
	def competitor_identification_task(self) -> Task:
		return Task(
			**self.tasks_config['competitor_identification_task'],
			output_pydantic=CompetitorTickerList
		)
	

	@crew
	def crew(self) -> Crew:
		"""Creates the CompetitionIdentifierCrew crew"""

		return Crew(
			agents=self.agents,
			tasks=self.tasks,
			process=Process.sequential,
			verbose=True
		)
