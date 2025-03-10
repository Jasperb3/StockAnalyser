import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from stock_analyser.utils.models import ExpertAnalystSignal
from stock_analyser.utils.constants import TIMESTAMP, REL_KNOW_DIR
from stock_analyser.tools.yfinance_buffett_analysis_tool import YFinanceBuffettAnalysisTool

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
class WarrenBuffetCrew():
	"""WarrenBuffetCrew crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'


	@agent
	def warren_buffet_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['warren_buffet_agent'],
			llm=gpt4_mini,
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
			llm=gemini_pro
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
