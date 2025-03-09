import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from stock_analyser.tools.google_search_tool import GoogleSearchTool
from stock_analyser.tools.yfinance_competitor_kpis_tool import YFinanceCompetitorKPIsTool
from stock_analyser.tools.yfinance_competitor_news_tool import YFinanceCompetitorNewsTool
from stock_analyser.tools.yfinance_stock_kpi_tool import YFinanceStockKPITool
from stock_analyser.tools.trafilatura_webscrape import TrafilaturaWebscrapeTool
from stock_analyser.utils.models import CompetitorList, CompetitorFinancialData, CompetitorSection
from stock_analyser.utils.constants import TIMESTAMP

from dotenv import load_dotenv

load_dotenv()

google_search_tool = GoogleSearchTool(api_key=os.getenv("GOOGLE_SEARCH_API_KEY"), cx=os.getenv("SEARCH_ENGINE_ID"))	

gemini_pro = LLM(
	model="gemini/gemini-2.0-pro-exp-02-05",
	api_key = os.getenv("GEMINI_API_KEY"),
	temperature=0.7,
	timeout=600
)

gemini_flash = LLM(
	model="gemini/gemini-2.0-flash",
	api_key = os.getenv("GEMINI_API_KEY"),
	temperature=0.7,
	timeout=600
)

gemini_flash_lite = LLM(
	model="gemini/gemini-2.0-flash-lite",
	api_key = os.getenv("GEMINI_API_KEY"),
	temperature=0.7,
	timeout=600
)

gemini_thinking = LLM(
	model="gemini/gemini-2.0-flash-thinking-exp-01-21",
	api_key = os.getenv("GEMINI_API_KEY"),
	temperature=0.7
)

gpt4_mini = LLM(
	model="gpt-4o-mini",
	api_key = os.getenv("OPENAI_API_KEY"),
	temperature=0.7
)

	
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
			tools=[google_search_tool, TrafilaturaWebscrapeTool()],
			llm=gpt4_mini,
			max_rpm=10
		)

	@agent
	def competitor_financial_data_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['competitor_financial_data_agent'],
			tools=[
				YFinanceCompetitorKPIsTool(),
				YFinanceCompetitorNewsTool(),
				YFinanceStockKPITool(),
				TrafilaturaWebscrapeTool()
			],
			llm=gpt4_mini,
			verbose=True,
		)
	
	@agent
	def competitor_section_writing_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['competitor_section_writing_agent'],
			llm=gemini_pro,
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
			output_file=f"knowledge/{TIMESTAMP}_competitor_financial_data.md"
		)
	
	@task
	def competitor_section_writing_task(self) -> Task:
		return Task(
			config=self.tasks_config['competitor_section_writing_task'],
			output_pydantic=CompetitorSection,
			output_file=f"knowledge/{TIMESTAMP}_competitor_section.md"
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
