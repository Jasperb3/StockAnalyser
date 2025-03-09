import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from stock_analyser.utils.models import ReportCritique, MarketIndustryContextModel
from stock_analyser.utils.constants import TIMESTAMP
from crewai_tools import EXASearchTool
from stock_analyser.tools.google_search_tool import GoogleSearchTool
from stock_analyser.tools.tavily_search import TavilySearchTool
from stock_analyser.tools.trafilatura_webscrape import TrafilaturaWebscrapeTool
from stock_analyser.tools.yfinance_news_tool import YFinanceNewsTool
from stock_analyser.tools.yfinance_company_info_tool import YFinanceCompanyInfoTool
from stock_analyser.tools.yfinance_industry_leaders_tool import YFinanceIndustryLeadersTool
from dotenv import load_dotenv

load_dotenv()

google_search_tool = GoogleSearchTool(api_key=os.getenv("GOOGLE_SEARCH_API_KEY"), cx=os.getenv("SEARCH_ENGINE_ID"))

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

	# Add knowledge_source as an __init__ parameter
	def __init__(self, knowledge_source=None):
		self.knowledge_source = knowledge_source

	# ----Agents----#
	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			llm=gpt4_mini,
			tools=[
				YFinanceCompanyInfoTool(),
				TavilySearchTool(),
				TrafilaturaWebscrapeTool(),
				exasearch_tool,
				google_search_tool,
				YFinanceNewsTool(),
				YFinanceIndustryLeadersTool(),
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
			output_file=f"knowledge/{TIMESTAMP}_Market_and_Industry_Context_research.md"
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
			output_file=f"knowledge/{TIMESTAMP}_Market_and_Industry_Context_section.md"
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the MarketAndIndustryCrew crew"""

		return Crew(
			agents=self.agents,
			tasks=self.tasks,
			process=Process.sequential,
			verbose=True,
			# Use the passed knowledge_source
			knowledge_sources=[self.knowledge_source] if self.knowledge_source else [],
			embedder={
				"provider": "google",
				"config": {
					"model": "models/text-embedding-004",
					"api_key": os.getenv("GEMINI_API_KEY")
				}
			}
		)
