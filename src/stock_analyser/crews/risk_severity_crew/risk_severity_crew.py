import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from stock_analyser.tools.google_search_tool import GoogleSearchTool
from stock_analyser.tools.trafilatura_webscrape import TrafilaturaWebscrapeTool
from stock_analyser.utils.models import RiskList, RiskSeverityList

from dotenv import load_dotenv

load_dotenv()

google_search_tool = GoogleSearchTool(api_key=os.getenv("GOOGLE_SEARCH_API_KEY"), cx=os.getenv("SEARCH_ENGINE_ID"))	

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
class RiskSeverityCrew():
	"""RiskSeverityCrew crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# ----Agents----#
	@agent
	def risk_identifier_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['risk_identifier_agent'],
			verbose=True,
			llm=gemini_pro,
			max_iter=5,
			max_rpm=3
		)

	@agent
	def risk_severity_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['risk_severity_agent'],
			tools=[TrafilaturaWebscrapeTool(), google_search_tool],
			llm=gemini_pro,
			verbose=True,
			max_iter=5
		)


	# ----Tasks----#
	@task
	def risk_identification_task(self) -> Task:
		return Task(
			config=self.tasks_config['risk_identification_task'],
			output_pydantic=RiskList
		)
	
	@task
	def risk_severity_task(self) -> Task:
		return Task(
			config=self.tasks_config['risk_severity_task'],
			output_pydantic=RiskSeverityList	
		)


	@crew
	def crew(self) -> Crew:
		"""Creates the RiskSeverityCrew crew"""

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True
		)