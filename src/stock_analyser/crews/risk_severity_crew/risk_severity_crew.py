import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from stock_analyser.tools.gemini_search_tool import GeminiSearchTool
from stock_analyser.tools.gemini_company_news_search_tool import CompanyNewsSearchTool
from stock_analyser.tools.trafilatura_webscrape import TrafilaturaWebscrapeTool
from stock_analyser.utils.models import RiskList, RiskSeverityList
from stock_analyser.utils.agent_llms import RESEARCH_MODEL, EXPERT_MODEL
from dotenv import load_dotenv

load_dotenv()

	
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
			llm=RESEARCH_MODEL,
			max_iter=5,
			max_rpm=3
		)

	@agent
	def risk_severity_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['risk_severity_agent'],
			tools=[TrafilaturaWebscrapeTool(), GeminiSearchTool(), CompanyNewsSearchTool()],
			llm=EXPERT_MODEL,
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