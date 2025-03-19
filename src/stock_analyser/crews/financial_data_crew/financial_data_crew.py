import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import QdrantVectorSearchTool
from stock_analyser.utils.embeddings_fn import custom_gemini_embedding_fn
from stock_analyser.utils.models import ReportCritique, FinancialData
from stock_analyser.utils.constants import TIMESTAMP, REL_KNOW_DIR
from stock_analyser.tools.calculator_tool import CalculatorTool
from stock_analyser.tools.yfinance_cash_flow_tool import YFinanceStockCashFlowTool
from stock_analyser.tools.yfinance_balance_sheet_tool import YFinanceStockBalanceSheetTool
from stock_analyser.tools.yfinance_financials_tool import YFinanceStockFinancialsTool
from stock_analyser.tools.yfinance_stock_kpi_tool import YFinanceStockKPITool
from stock_analyser.tools.yfinance_income_tool import YFinanceIncomeTool
from stock_analyser.tools.yfinance_analysis_and_holdings_tool import YFinanceAnalysisAndHoldingsTool
from stock_analyser.utils.agent_llms import RESEARCH_MODEL, WRITING_MODEL, CRITIC_MODEL, EDITOR_MODEL

from dotenv import load_dotenv

load_dotenv()

qdrant_tool = QdrantVectorSearchTool(
    qdrant_url=os.getenv("QDRANT_CLUSTER_URL"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    limit=5,
    score_threshold=0.35,
	custom_embedding_fn=custom_gemini_embedding_fn,
	description = "A tool to search the SEC filings."
)


@CrewBase
class FinancialDataCrew():
	"""FinancialDataCrew crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'


	# ----Agents----#
	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			llm=RESEARCH_MODEL,
			tools=[
				YFinanceStockKPITool(),
				YFinanceIncomeTool(),
				YFinanceStockCashFlowTool(),
				YFinanceStockBalanceSheetTool(),
				YFinanceStockFinancialsTool(),
				YFinanceAnalysisAndHoldingsTool(),
				CalculatorTool(),
				# FilingsSearchTool(
				# 	config=dict(
				# 		llm=dict(
				# 			provider="google",
				# 			config=dict(
				# 				model="gemini/gemini-2.0-pro-exp-02-05"
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
	def comprehensive_financial_analysis_task(self) -> Task:
		return Task(
			config=self.tasks_config['comprehensive_financial_analysis_task'],
			output_pydantic=FinancialData,
			output_file=f"{REL_KNOW_DIR}/{TIMESTAMP}_Financial_Data_complete.md",
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
			output_file=f"{REL_KNOW_DIR}/{TIMESTAMP}_Financial_Data_section.md"
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the FinancialDataCrew crew"""

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True
		)
