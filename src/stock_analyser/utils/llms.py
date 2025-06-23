import os
from crewai import LLM


gemini_pro = LLM(
	model="gemini/gemini-2.0-pro-exp-02-05",
	api_key = os.getenv("GEMINI_API_KEY"),
	temperature=0.7,
	timeout=600
)

gemini_flash = LLM(
	model="gemini/gemini-2.5-flash-preview-04-17",
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

gpt4o_mini = LLM(
	model="gpt-4o-mini",
	api_key = os.getenv("OPENAI_API_KEY"),
	temperature=0.7
)

gpt4o = LLM(
	model="gpt-4o",
	api_key = os.getenv("OPENAI_API_KEY"),
	temperature=0.7
)

gpt41 = LLM(
	model="gpt-4.1",
	api_key = os.getenv("OPENAI_API_KEY"),
	temperature=0.7
)

gpt41mini = LLM(
	model="gpt-4.1-mini",
	api_key = os.getenv("OPENAI_API_KEY"),
	temperature=0.7
)

gpt41nano = LLM(
	model="gpt-4.1-nano",
	api_key = os.getenv("OPENAI_API_KEY"),
	temperature=0.7
)

gemma3 = LLM(
	model="ollama/gemma3:12b",
    base_url="http://localhost:11434",
	api_key = 'NA',
	temperature=0.7
)

mistral_large = LLM(
	model="mistral/mistral-large-latest",
	api_key = os.getenv("MISTRAL_API_KEY"),
	temperature=0.25
)

llama_4_maverick = LLM(
	model="groq/meta-llama/llama-4-maverick-17b-128e-instruct",
	api_key = os.getenv("GROQ_API_KEY"),
	temperature=0.7
)

llama_4_scout = LLM(
	model="groq/meta-llama/llama-4-scout-17b-16e-instruct",
	api_key = os.getenv("GROQ_API_KEY"),
	temperature=0.7
)

