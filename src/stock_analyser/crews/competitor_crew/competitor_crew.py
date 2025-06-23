import os
from stock_analyser.utils.models import CompetitorResearch
from stock_analyser.utils.constants import TIMESTAMP, REL_KNOW_DIR
from stock_analyser.utils.agent_llms import RESEARCH_MODEL, EXPERT_MODEL
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from stock_analyser.tools.yfinance_competitor_financial_metrics_tool import YFinanceCompetitorFinancialMetricsTool
from stock_analyser.tools.trafilatura_webscrape import TrafilaturaWebscrapeTool
from stock_analyser.tools.google_search_tool import GoogleSearchTool
from stock_analyser.tools.yfinance_competitor_news_tool import YFinanceCompetitorNewsTool
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
	def competitor_research_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['competitor_research_agent'],
			tools=[
				YFinanceCompetitorFinancialMetricsTool(),
				YFinanceCompetitorNewsTool(),
				TrafilaturaWebscrapeTool(),
				google_search_tool,
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
	def competitor_research_task(self) -> Task:
		return Task(
			config=self.tasks_config['competitor_research_task'],
			output_pydantic=CompetitorResearch
		)
	
	@task
	def competitor_section_writing_task(self) -> Task:
		return Task(
			config=self.tasks_config['competitor_section_writing_task'],
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
