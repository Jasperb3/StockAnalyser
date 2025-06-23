import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from stock_analyser.utils.models import ReportCritique, MarketIndustryContextModel
from stock_analyser.utils.constants import TIMESTAMP, REL_KNOW_DIR
from stock_analyser.utils.agent_llms import RESEARCH_MODEL, WRITING_MODEL, CRITIC_MODEL, EDITOR_MODEL
from crewai_tools import EXASearchTool
from stock_analyser.tools.gemini_search_tool import GeminiSearchTool
from stock_analyser.tools.qdrant_sec_filings_search_tool import QdrantSECFilingsSearchTool
from stock_analyser.tools.gemini_company_news_search_tool import CompanyNewsSearchTool
from stock_analyser.tools.tavily_search import TavilySearchTool
from stock_analyser.tools.trafilatura_webscrape import TrafilaturaWebscrapeTool
from stock_analyser.tools.yfinance_company_info_tool import YFinanceCompanyInfoTool
from stock_analyser.tools.yfinance_industry_leaders_tool import YFinanceIndustryLeadersTool
from stock_analyser.tools.linkup_search_tool import LinkUpSearchTool
from dotenv import load_dotenv

load_dotenv()


qdrant_sec_filings_tool = QdrantSECFilingsSearchTool(
	qdrant_url=os.getenv("QDRANT_CLUSTER_URL"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    collection_name=os.getenv("QDRANT_COLLECTION_NAME")
)

exa_api_key = os.getenv("EXA_API_KEY")
exasearch_tool = EXASearchTool(api_key=exa_api_key, content=True, summary=True)


@CrewBase
class MarketAndIndustryCrew():
	"""MarketAndIndustryCrew crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'


	# ----Agents----#
	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			llm=RESEARCH_MODEL,
			tools=[
				qdrant_sec_filings_tool,
				YFinanceCompanyInfoTool(),
				GeminiSearchTool(),
				CompanyNewsSearchTool(),
				YFinanceIndustryLeadersTool(),
				TavilySearchTool(),
				TrafilaturaWebscrapeTool(),
				LinkUpSearchTool()
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

	@agent
	def critic(self) -> Agent:
		return Agent(
			config=self.agents_config['critic'],
			llm=CRITIC_MODEL,
			verbose=True
		)

	@agent
	def editor(self) -> Agent:
		return Agent(
			config=self.agents_config['editor'],
			llm=EDITOR_MODEL,
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
