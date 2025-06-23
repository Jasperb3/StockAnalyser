import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from stock_analyser.utils.models import NewsAndResearchModel
from stock_analyser.utils.constants import TIMESTAMP, REL_KNOW_DIR
from stock_analyser.utils.agent_llms import RESEARCH_MODEL, WRITING_MODEL
from crewai_tools import EXASearchTool
from stock_analyser.tools.google_search_tool import GoogleSearchTool
from stock_analyser.tools.tavily_search import TavilySearchTool
from stock_analyser.tools.trafilatura_webscrape import TrafilaturaWebscrapeTool
from stock_analyser.tools.yfinance_news_tool import YFinanceNewsTool
from dotenv import load_dotenv

load_dotenv()

exa_api_key = os.getenv("EXA_API_KEY")
exasearch_tool = EXASearchTool(api_key=exa_api_key, content=True, summary=True)

google_search_tool = GoogleSearchTool(api_key = os.getenv("GOOGLE_SEARCH_API_KEY"), cx = os.getenv("SEARCH_ENGINE_ID"))


@CrewBase
class NewsAndResearchCrew():
	"""NewsAndResearchCrew crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# ----Agents----#
	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			llm=RESEARCH_MODEL,
			tools=[
				# TavilySearchTool(),
				YFinanceNewsTool(),
				# TrafilaturaWebscrapeTool(),
				exasearch_tool,
				# google_search_tool,
			],
			verbose=True,
			max_iter=10,
			# max_rpm=10
		)

	@agent
	def writer(self) -> Agent:
		return Agent(
			config=self.agents_config['writer'],
			llm=WRITING_MODEL,
			verbose=True
		)


	# ----Tasks----#
	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
			output_pydantic=NewsAndResearchModel,
			output_file=f"{REL_KNOW_DIR}/{TIMESTAMP}_News_and_Research_research.md"
		)

	@task
	def writing_task(self) -> Task:
		return Task(
			config=self.tasks_config['writing_task'],
			output_file=f"{REL_KNOW_DIR}/{TIMESTAMP}_News_and_Research_section.md"
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the NewsAndResearchCrew crew"""

		return Crew(
			agents=self.agents,
			tasks=self.tasks,
			process=Process.sequential,
			verbose=True
		)
