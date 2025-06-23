import os
from stock_analyser.utils.constants import TIMESTAMP, REL_KNOW_DIR
from stock_analyser.utils.agent_llms import RESEARCH_MODEL, WRITING_MODEL, CRITIC_MODEL, EDITOR_MODEL
from stock_analyser.utils.models import ReportCritique, AnalystsInsightsModel
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import EXASearchTool
from stock_analyser.tools.calculator_tool import CalculatorTool
from stock_analyser.tools.gemini_search_tool import GeminiSearchTool
from stock_analyser.tools.yfinance_analysis_and_holdings_tool import YFinanceAnalysisAndHoldingsTool
from stock_analyser.tools.yfinance_income_tool import YFinanceIncomeTool
from stock_analyser.tools.yfinance_swing_trading_tool import YFinanceSwingTradingTool
from stock_analyser.tools.linkup_search_tool import LinkUpSearchTool

from dotenv import load_dotenv

load_dotenv()

exa_api_key = os.getenv("EXA_API_KEY")
exasearch_tool = EXASearchTool(api_key=exa_api_key, content=True, summary=True)


@CrewBase
class AnalystInsightsCrew():
	"""AnalystInsightsCrew crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'


	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			llm=RESEARCH_MODEL,
			tools=[
				YFinanceAnalysisAndHoldingsTool(),
				YFinanceIncomeTool(),
				YFinanceSwingTradingTool(),
				CalculatorTool(),
				LinkUpSearchTool(),
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
			max_iter=7,
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
			tools=[
				GeminiSearchTool()
			],
			llm=EDITOR_MODEL,
			verbose=True
		)

	# ----Tasks----#
	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
			output_file=f"{REL_KNOW_DIR}/{TIMESTAMP}_Analyst_Insights_research.md",
			output_pydantic=AnalystsInsightsModel
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
			output_file=f"{REL_KNOW_DIR}/{TIMESTAMP}_Analyst_Insights_section.md"
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the AnalystInsightsCrew crew"""
		

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True
		)
