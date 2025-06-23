import os
from stock_analyser.utils.agent_llms import EMAIL_MODEL, WRITING_MODEL
from stock_analyser.utils.models import Email
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from stock_analyser.tools.gmail_tool_with_attachment import GmailAttachmentTool
from dotenv import load_dotenv

load_dotenv()


@CrewBase
class GmailAttachmentCrew():
	"""GmailAttachmentCrew crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def email_writing_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['email_writing_agent'],
			llm=WRITING_MODEL,
			verbose=True
)

	@agent
	def gmail_draft_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['gmail_draft_agent'],
			tools=[GmailAttachmentTool()],
			llm=EMAIL_MODEL,
			verbose=True
)

	@task
	def email_writing_task(self) -> Task:
		return Task(
			config=self.tasks_config['email_writing_task'],
			output_pydantic=Email
		)

	@task
	def gmail_draft_task(self) -> Task:
		return Task(
			config=self.tasks_config['gmail_draft_task']
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the GmailAttachmentCrew crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
