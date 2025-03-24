from stock_analyser.utils.llms import (
    gpt4o_mini,
    gemini_pro,
    gemini_flash,
    gemini_thinking,
    mistral_large,
    gpt4o
)

RESEARCH_MODEL = mistral_large

WRITING_MODEL = gemini_thinking

EXPERT_MODEL = gpt4o

CRITIC_MODEL = gemini_pro

EDITOR_MODEL = gemini_pro

EXPERT_ANALYSIS_MODEL = gpt4o_mini

EMAIL_MODEL = gpt4o_mini
