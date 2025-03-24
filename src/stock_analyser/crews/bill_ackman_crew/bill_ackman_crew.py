import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from stock_analyser.utils.models import ExpertAnalystSignal
from stock_analyser.utils.constants import TIMESTAMP, REL_KNOW_DIR
from stock_analyser.utils.agent_llms import EXPERT_ANALYSIS_MODEL, WRITING_MODEL
from stock_analyser.tools.yfinance_ackman_analysis_tool import YFinanceAckmanAnalysisTool

from dotenv import load_dotenv

load_dotenv()


@CrewBase
class BillAckmanCrew():
	"""BillAckmanCrew crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'


	# ----Agents----#
	@agent
	def bill_ackman_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['bill_ackman_agent'],
			llm=EXPERT_ANALYSIS_MODEL,
			tools=[
				YFinanceAckmanAnalysisTool()
			],
			verbose=True,
			max_rpm=10
		)
	
	@agent
	def bill_ackman_writer(self) -> Agent:
		return Agent(
			config=self.agents_config['bill_ackman_writer'],
			llm=WRITING_MODEL,
			verbose=True
		)

	# ----Tasks----#
	@task
	def bill_ackman_analysis_task(self) -> Task:
		return Task(
			config=self.tasks_config['bill_ackman_analysis_task'],
			output_pydantic=ExpertAnalystSignal
		)
	
	@task
	def bill_ackman_writeup_task(self) -> Task:
		return Task(
			config=self.tasks_config['bill_ackman_writeup_task'],
			output_file=f"{REL_KNOW_DIR}/{TIMESTAMP}_Bill_Ackman_writeup.md"
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the BillAckmanCrew"""
		

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True
		)
