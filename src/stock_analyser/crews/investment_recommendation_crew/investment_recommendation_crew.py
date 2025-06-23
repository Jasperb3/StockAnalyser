from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from stock_analyser.utils.models import ReportCritique, ResearchQuestions, ResearchAnswers
from stock_analyser.utils.constants import TIMESTAMP, REL_KNOW_DIR
from stock_analyser.utils.agent_llms import RESEARCH_MODEL, WRITING_MODEL, CRITIC_MODEL, EDITOR_MODEL, GPT4O
from stock_analyser.tools.gemini_search_tool import GeminiSearchTool
from stock_analyser.tools.yfinance_swing_trading_tool import YFinanceSwingTradingTool
from stock_analyser.tools.linkup_search_tool import LinkUpSearchTool
from dotenv import load_dotenv

load_dotenv()


@CrewBase
class InvestmentRecommendationCrew():
	"""InvestmentRecommendationCrew crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'


	# ----Agents----#
	@agent
	def planner(self) -> Agent:
		return Agent(
			config=self.agents_config['planner'],
			llm=GPT4O,
			verbose=True
		)

	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			llm=RESEARCH_MODEL,
			verbose=True,
			tools=[
				GeminiSearchTool(),
				YFinanceSwingTradingTool(),
				LinkUpSearchTool()
			],
			max_iter=10
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
	def planning_task(self) -> Task:
		return Task(
			config=self.tasks_config['planning_task'],
			output_pydantic=ResearchQuestions,
		)
	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
			output_pydantic=ResearchAnswers,
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
			output_file=f"{REL_KNOW_DIR}/{TIMESTAMP}_Investment_Recommendation_section.md"
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the InvestmentRecommendationCrew crew"""


		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True
		)
