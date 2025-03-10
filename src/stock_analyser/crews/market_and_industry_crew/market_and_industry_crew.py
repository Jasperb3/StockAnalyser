import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from stock_analyser.utils.models import ReportCritique, MarketIndustryContextModel
from stock_analyser.utils.constants import TIMESTAMP, REL_KNOW_DIR
from crewai_tools import EXASearchTool
from stock_analyser.tools.gemini_search_tool import GeminiSearchTool
from stock_analyser.tools.gemini_company_news_search_tool import CompanyNewsSearchTool
from stock_analyser.tools.tavily_search import TavilySearchTool
from stock_analyser.tools.trafilatura_webscrape import TrafilaturaWebscrapeTool
from stock_analyser.tools.yfinance_news_tool import YFinanceNewsTool
from stock_analyser.tools.yfinance_company_info_tool import YFinanceCompanyInfoTool
from stock_analyser.tools.yfinance_industry_leaders_tool import YFinanceIndustryLeadersTool
from dotenv import load_dotenv

load_dotenv()

exa_api_key = os.getenv("EXA_API_KEY")
exasearch_tool = EXASearchTool(api_key=exa_api_key, content=True, summary=True)


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
class MarketAndIndustryCrew():
	"""MarketAndIndustryCrew crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# Add knowledge_source and qdrant_tool as __init__ parameters
	def __init__(self, qdrant_tool=None):
		self.qdrant_tool = qdrant_tool

	# ----Agents----#
	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			llm=gpt4_mini,
			tools=[
				YFinanceCompanyInfoTool(),
				GeminiSearchTool(),
				CompanyNewsSearchTool(),
				YFinanceIndustryLeadersTool(),
				TavilySearchTool(),
				TrafilaturaWebscrapeTool(),
				YFinanceNewsTool(),
				# exasearch_tool,
				self.qdrant_tool
			],
			verbose=True,
			max_iter=10,
			# max_rpm=10
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
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
			output_pydantic=MarketIndustryContextModel,
			output_file=f"{REL_KNOW_DIR}/{TIMESTAMP}_Market_and_Industry_Context_research.md"
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
			output_file=f"{REL_KNOW_DIR}/{TIMESTAMP}_Market_and_Industry_Context_section.md"
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the MarketAndIndustryCrew crew"""

		return Crew(
			agents=self.agents,
			tasks=self.tasks,
			process=Process.sequential,
			verbose=True
		)
