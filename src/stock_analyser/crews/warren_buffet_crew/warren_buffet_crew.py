from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from stock_analyser.utils.models import ExpertAnalystSignal
from stock_analyser.utils.constants import TIMESTAMP, REL_KNOW_DIR
from stock_analyser.utils.agent_llms import EXPERT_ANALYSIS_MODEL, WRITING_MODEL
from stock_analyser.tools.yfinance_buffett_analysis_tool import YFinanceBuffettAnalysisTool

from dotenv import load_dotenv

load_dotenv()


@CrewBase
class WarrenBuffetCrew():
	"""WarrenBuffetCrew crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'


	@agent
	def warren_buffet_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['warren_buffet_agent'],
			llm=EXPERT_ANALYSIS_MODEL,
			tools=[
				YFinanceBuffettAnalysisTool()
			],
			verbose=True,
			max_rpm=10
		)
	
	@agent
	def warren_buffet_writer(self) -> Agent:
		return Agent(
			config=self.agents_config['warren_buffet_writer'],
			llm=WRITING_MODEL,
			verbose=True
		)

	# ----Tasks----#
	@task
	def warren_buffet_analysis_task(self) -> Task:
		return Task(
			config=self.tasks_config['warren_buffet_analysis_task'],
			output_pydantic=ExpertAnalystSignal
		)
	
	@task
	def warren_buffet_writeup_task(self) -> Task:
		return Task(
			config=self.tasks_config['warren_buffet_writeup_task'],
			output_file=f"{REL_KNOW_DIR}/{TIMESTAMP}_Warren_Buffet_writeup.md"
		)
	

	@crew
	def crew(self) -> Crew:
		"""Creates the WarrenBuffetCrew crew"""
		

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True
		)
