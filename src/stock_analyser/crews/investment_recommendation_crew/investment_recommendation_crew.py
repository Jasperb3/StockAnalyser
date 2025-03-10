import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from stock_analyser.utils.models import ReportCritique, ResearchQuestions, ResearchAnswers
from stock_analyser.utils.constants import TIMESTAMP, REL_KNOW_DIR
from stock_analyser.tools.gemini_search_tool import GeminiSearchTool
from stock_analyser.tools.gemini_company_news_search_tool import CompanyNewsSearchTool
from stock_analyser.tools.tavily_search import TavilySearchTool
from stock_analyser.tools.trafilatura_webscrape import TrafilaturaWebscrapeTool
from stock_analyser.tools.yfinance_news_tool import YFinanceNewsTool
from stock_analyser.tools.yfinance_stock_kpi_tool import YFinanceStockKPITool
from stock_analyser.tools.yfinance_financials_tool import YFinanceStockFinancialsTool


from dotenv import load_dotenv
load_dotenv()

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
class InvestmentRecommendationCrew():
	"""InvestmentRecommendationCrew crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'


	# ----Agents----#
	@agent
	def planner(self) -> Agent:
		return Agent(
			config=self.agents_config['planner'],
			llm=gemini_pro,
			verbose=True
		)

	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			llm=gpt4_mini,
			verbose=True,
			tools=[
				GeminiSearchTool(),
				CompanyNewsSearchTool(),
				YFinanceStockKPITool(),
				YFinanceStockFinancialsTool(),
				YFinanceNewsTool(),
				TrafilaturaWebscrapeTool(),
				TavilySearchTool()
			],
			max_iter=10
		)

	@agent
	def writer(self) -> Agent:
		return Agent(
			config=self.agents_config['writer'],
			llm=gemini_thinking,
			verbose=True
		)

	@agent
	def critic(self) -> Agent:
		return Agent(
			config=self.agents_config['critic'],
			llm=gemini_pro,
			verbose=True
		)

	@agent
	def editor(self) -> Agent:
		return Agent(
			config=self.agents_config['editor'],
			llm=gemini_pro,
			verbose=True
		)

	# ----Tasks----#
	@task
	def planning_task(self) -> Task:
		return Task(
			config=self.tasks_config['planning_task'],
			output_pydantic=ResearchQuestions,
		)
	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
			output_pydantic=ResearchAnswers,
		)

	@task
	def writing_task(self) -> Task:
		return Task(
			config=self.tasks_config['writing_task'],
		)

	@task
	def critic_task(self) -> Task:
		return Task(
			config=self.tasks_config['critic_task'],
			output_pydantic=ReportCritique
		)

	@task
	def editing_task(self) -> Task:
		return Task(
			config=self.tasks_config['editing_task'],
			output_file=f"{REL_KNOW_DIR}/{TIMESTAMP}_Investment_Recommendation_section.md"
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the InvestmentRecommendationCrew crew"""


		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True
		)
