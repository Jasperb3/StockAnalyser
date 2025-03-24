import yfinance as yf
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, before_kickoff
from stock_analyser.utils.convert_currency import convert_currency
from stock_analyser.utils.constants import TIMESTAMP, REL_KNOW_DIR
from stock_analyser.utils.agent_llms import WRITING_MODEL

from dotenv import load_dotenv

load_dotenv()

@CrewBase
class TrendsCrew():
	"""TrendsCrew crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@before_kickoff
	def prepare_inputs(self, inputs):
		stock = yf.Ticker(inputs['company_ticker'])

		exchange_rate = convert_currency(inputs['company_ticker'])

		financial_data = stock.financials
		if financial_data is not None and not financial_data.empty:
			financial_data = financial_data.apply(lambda x: x * exchange_rate)

		try:
			revenue_data = financial_data.loc['Total Revenue'].to_dict()
			revenue_data = {key.strftime('%Y-%m-%d'): value for key, value in revenue_data.items()}
			inputs['revenue_data'] = revenue_data
		except Exception as e:
			print(f"Error converting revenue data for {inputs['company_ticker']}: {e}")
			inputs['revenue_data'] = None

		try:
			net_income_data = financial_data.loc['Net Income'].to_dict()
			net_income_data = {key.strftime('%Y-%m-%d'): value for key, value in net_income_data.items()}
			inputs['net_income_data'] = net_income_data
		except Exception as e:
			print(f"Error converting net income data for {inputs['company_ticker']}: {e}")
			inputs['net_income_data'] = None

		try:
			gross_profit_data = financial_data.loc['Gross Profit']
			revenue_data = financial_data.loc['Total Revenue']
			gross_margin_data = gross_profit_data / revenue_data
			gross_margin_data = {key.strftime('%Y-%m-%d'): value for key, value in gross_margin_data.to_dict().items()}
			inputs['gross_margin_data'] = gross_margin_data
		except Exception as e:
			print(f"Error converting gross margin data for {inputs['company_ticker']}: {e}")
			inputs['gross_margin_data'] = None

		try:
			diluted_eps_data = financial_data.loc['Diluted EPS'].to_dict()
			diluted_eps_data = {key.strftime('%Y-%m-%d'): value for key, value in diluted_eps_data.items()}
			inputs['eps_data'] = diluted_eps_data
		except Exception as e:
			print(f"Error converting diluted EPS data for {inputs['company_ticker']}: {e}")
			inputs['eps_data'] = None

		cash_flow_data = stock.cashflow
		if cash_flow_data is not None and not cash_flow_data.empty:
			cash_flow_data = cash_flow_data.apply(lambda x: x * exchange_rate)

		try:
			free_cash_flow_data = cash_flow_data.loc['Free Cash Flow'].to_dict()
			free_cash_flow_data = {key.strftime('%Y-%m-%d'): value for key, value in free_cash_flow_data.items()}
			inputs['free_cash_flow_data'] = free_cash_flow_data
		except Exception as e:
			print(f"Error converting free cash flow data for {inputs['company_ticker']}: {e}")
			inputs['free_cash_flow_data'] = None

		return inputs

	# ----Agents----#
	@agent
	def trend_analysist(self) -> Agent:
		return Agent(
			config=self.agents_config['trend_analysist'],
			llm=WRITING_MODEL,
			verbose=True,
			max_iter=10,
			# max_rpm=10
		)


	# ----Tasks----#
	@task
	def trend_analysis_task(self) -> Task:
		return Task(
			config=self.tasks_config['trend_analysis_task'],
			output_file=f"{REL_KNOW_DIR}/{TIMESTAMP}_Trends_research.md"
		)


	@crew
	def crew(self) -> Crew:
		"""Creates the TrendsCrew crew"""

		return Crew(
			agents=self.agents,
			tasks=self.tasks,
			process=Process.sequential,
			verbose=True
		)
