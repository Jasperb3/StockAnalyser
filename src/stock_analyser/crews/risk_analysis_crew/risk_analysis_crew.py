import os
from pathlib import Path
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from stock_analyser.utils.embeddings_fn import custom_gemini_embedding_fn
from stock_analyser.utils.models import ReportCritique, RiskAnalysisModel
from stock_analyser.utils.constants import TIMESTAMP, REL_KNOW_DIR
from stock_analyser.utils.agent_llms import RESEARCH_MODEL, WRITING_MODEL, CRITIC_MODEL, EDITOR_MODEL
from crewai_tools import EXASearchTool#, QdrantVectorSearchTool
from stock_analyser.tools.qdrant_sec_filings_search_tool import QdrantSECFilingsSearchTool
from stock_analyser.tools.gemini_search_tool import GeminiSearchTool
from stock_analyser.tools.gemini_company_news_search_tool import CompanyNewsSearchTool
from stock_analyser.tools.tavily_search import TavilySearchTool
from stock_analyser.tools.trafilatura_webscrape import TrafilaturaWebscrapeTool
from stock_analyser.tools.yfinance_news_tool import YFinanceNewsTool
from stock_analyser.tools.yfinance_esg_tool import YFinanceESGTool
from stock_analyser.tools.filings_search_tool import FilingsSearchTool
from stock_analyser.tools.calculator_tool import CalculatorTool

from dotenv import load_dotenv

load_dotenv()

# google_search_tool = GoogleSearchTool(api_key=os.getenv("GOOGLE_SEARCH_API_KEY"), cx=os.getenv("SEARCH_ENGINE_ID"))

# qdrant_tool = QdrantVectorSearchTool(
#     qdrant_url=os.getenv("QDRANT_CLUSTER_URL"),
#     qdrant_api_key=os.getenv("QDRANT_API_KEY"),
#     collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
#     limit=5,
#     score_threshold=0.35,
# 	custom_embedding_fn=custom_gemini_embedding_fn,
# 	description = "Use this tool to perform a vector search of the SEC filings."
# )

qdrant_sec_filings_tool = QdrantSECFilingsSearchTool(
	qdrant_url=os.getenv("QDRANT_CLUSTER_URL"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    collection_name=os.getenv("QDRANT_COLLECTION_NAME")
)

exa_api_key = os.getenv("EXA_API_KEY")
exasearch_tool = EXASearchTool(api_key=exa_api_key, content=True, summary=True)


@CrewBase
class RiskAnalysisCrew():
	"""RiskAnalysisCrew crew"""

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
				GeminiSearchTool(),
				CompanyNewsSearchTool(),
				YFinanceNewsTool(),
				YFinanceESGTool(),
				TavilySearchTool(),
				TrafilaturaWebscrapeTool(),
				exasearch_tool,
				CalculatorTool(),
				# FilingsSearchTool(
				# 	config=dict(
				# 		llm=dict(
				# 			provider="google",
				# 			config=dict(
				# 				model="gemini/gemini-2.0-flash-exp"
				# 			),
				# 		),
				# 		embedder={
				# 			"provider": "google",
				# 			"config": {
				# 				"model": "models/text-embedding-004"
				# 			}
				# 		}
				# 	),
				# 	directory=str(Path(__file__).parent.parent.parent.parent.parent / "filings")
				# )
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
			output_pydantic=RiskAnalysisModel,
			output_file=f"{REL_KNOW_DIR}/{TIMESTAMP}_Risk_Analysis_research.md"
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
			output_file=f"{REL_KNOW_DIR}/{TIMESTAMP}_Risk_Analysis_section.md"
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the RiskAnalysisCrew crew"""
		

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True
		)


