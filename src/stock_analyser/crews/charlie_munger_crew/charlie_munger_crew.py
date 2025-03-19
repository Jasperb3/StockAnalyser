import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from stock_analyser.utils.models import ExpertAnalystSignal
from stock_analyser.utils.constants import TIMESTAMP, REL_KNOW_DIR
from stock_analyser.utils.agent_llms import RESEARCH_MODEL, WRITING_MODEL
from stock_analyser.tools.yfinance_munger_analysis_tool import YFinanceMungerAnalysisTool

from dotenv import load_dotenv

load_dotenv()


@CrewBase
class CharlieMungerCrew():
	"""CharlieMungerCrew crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'


	# ----Agents----#
	@agent
	def charlie_munger_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['charlie_munger_agent'],
			llm=RESEARCH_MODEL,
			tools=[
				YFinanceMungerAnalysisTool()
			],
			verbose=True,
			max_rpm=10
		)
	
	@agent
	def charlie_munger_writer(self) -> Agent:
		return Agent(
			config=self.agents_config['charlie_munger_writer'],
			llm=WRITING_MODEL,
			verbose=True
		)

	# ----Tasks----#
	@task
	def charlie_munger_analysis_task(self) -> Task:
		return Task(
			config=self.tasks_config['charlie_munger_analysis_task'],
			output_pydantic=ExpertAnalystSignal
		)
	
	@task
	def charlie_munger_writeup_task(self) -> Task:
		return Task(
			config=self.tasks_config['charlie_munger_writeup_task'],
			output_file=f"{REL_KNOW_DIR}/{TIMESTAMP}_Charlie_Munger_writeup.md"
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the CharlieMungerCrew"""
		

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True
		)
