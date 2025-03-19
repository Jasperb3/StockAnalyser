import os
from stock_analyser.utils.models import CompetitorList, CompetitorFinancialData, CompetitorSection
from stock_analyser.utils.constants import TIMESTAMP, REL_KNOW_DIR
from stock_analyser.utils.agent_llms import RESEARCH_MODEL, EXPERT_MODEL
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from stock_analyser.tools.gemini_search_tool import GeminiSearchTool
from stock_analyser.tools.yfinance_competitor_kpis_tool import YFinanceCompetitorKPIsTool
from stock_analyser.tools.gemini_competitor_news_search_tool import CompetitorNewsSearchTool
from stock_analyser.tools.yfinance_stock_kpi_tool import YFinanceStockKPITool
from stock_analyser.tools.trafilatura_webscrape import TrafilaturaWebscrapeTool
from stock_analyser.tools.tavily_search import TavilySearchTool
from stock_analyser.tools.google_search_tool import GoogleSearchTool
from stock_analyser.tools.yfinance_competitor_news_tool import YFinanceCompetitorNewsTool
from stock_analyser.tools.yfinance_industry_leaders_tool import YFinanceIndustryLeadersTool
from dotenv import load_dotenv

load_dotenv()

google_search_tool = GoogleSearchTool(api_key=os.getenv("GOOGLE_SEARCH_API_KEY"), cx=os.getenv("SEARCH_ENGINE_ID"))

	
@CrewBase
class CompetitorCrew():
	"""CompetitorCrew crew"""

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
				TrafilaturaWebscrapeTool()
			],
			llm=RESEARCH_MODEL
		)

	@agent
	def competitor_financial_data_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['competitor_financial_data_agent'],
			tools=[
				YFinanceCompetitorKPIsTool(),
				YFinanceCompetitorNewsTool(),
				TrafilaturaWebscrapeTool(),
				google_search_tool,
				# CompetitorNewsSearchTool()
			],
			llm=RESEARCH_MODEL,
			verbose=True,
		)
	
	@agent
	def competitor_section_writing_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['competitor_section_writing_agent'],
			llm=EXPERT_MODEL,
			verbose=True,
		)

	# ----Tasks----#
	@task
	def competitor_identification_task(self) -> Task:
		return Task(
			config=self.tasks_config['competitor_identification_task'],
			output_pydantic=CompetitorList
		)
	
	@task
	def competitor_financial_data_task(self) -> Task:
		return Task(
			config=self.tasks_config['competitor_financial_data_task'],
			output_pydantic=CompetitorFinancialData,
			output_file=f"{REL_KNOW_DIR}/{TIMESTAMP}_competitor_financial_data.md"
		)
	
	@task
	def competitor_section_writing_task(self) -> Task:
		return Task(
			config=self.tasks_config['competitor_section_writing_task'],
			output_pydantic=CompetitorSection,
			output_file=f"{REL_KNOW_DIR}/{TIMESTAMP}_competitor_section.md"
		)


	@crew
	def crew(self) -> Crew:
		"""Creates the CompetitorCrew crew"""

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True
		)
