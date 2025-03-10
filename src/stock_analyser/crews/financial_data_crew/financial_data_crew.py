import os
from pathlib import Path
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from stock_analyser.utils.models import ReportCritique, FinancialData
from stock_analyser.utils.constants import TIMESTAMP, REL_KNOW_DIR
from stock_analyser.tools.calculator_tool import CalculatorTool
from stock_analyser.tools.yfinance_cash_flow_tool import YFinanceStockCashFlowTool
from stock_analyser.tools.yfinance_balance_sheet_tool import YFinanceStockBalanceSheetTool
from stock_analyser.tools.yfinance_financials_tool import YFinanceStockFinancialsTool
from stock_analyser.tools.yfinance_stock_kpi_tool import YFinanceStockKPITool
from stock_analyser.tools.yfinance_income_tool import YFinanceIncomeTool
from stock_analyser.tools.filings_search_tool import FilingsSearchTool

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
class FinancialDataCrew():
	"""FinancialDataCrew crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	def __init__(self, qdrant_tool=None):
		self.qdrant_tool = qdrant_tool


	# ----Agents----#
	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			llm=gpt4_mini,
			tools=[

				# Add Qdrant tool if available

				self.qdrant_tool,

				
				YFinanceStockKPITool(),
				YFinanceIncomeTool(),
				YFinanceStockCashFlowTool(),
				YFinanceStockBalanceSheetTool(),
				YFinanceStockFinancialsTool(),
				CalculatorTool(),
				FilingsSearchTool(
					config=dict(
						llm=dict(
							provider="google",
							config=dict(
								model="gemini/gemini-2.0-pro-exp-02-05"
							),
						),
						embedder={
							"provider": "google",
							"config": {
								"model": "models/text-embedding-004"
							}
						}
					),
					directory=str(Path(__file__).parent.parent.parent.parent.parent / "filings")
				)
			

			] if self.qdrant_tool else [

				
				YFinanceStockKPITool(),
				YFinanceIncomeTool(),
				YFinanceStockCashFlowTool(),
				YFinanceStockBalanceSheetTool(),
				YFinanceStockFinancialsTool(),
				CalculatorTool(),
				FilingsSearchTool(
					config=dict(
						llm=dict(
							provider="google",
							config=dict(
								model="gemini/gemini-2.0-pro-exp-02-05"
							),
						),
						embedder={
							"provider": "google",
							"config": {
								"model": "models/text-embedding-004"
							}
						}
					),
					directory=str(Path(__file__).parent.parent.parent.parent.parent / "filings")
				)
			

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
			llm=gemini_flash,
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
