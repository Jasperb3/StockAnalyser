from typing import Type

from crewai.tools import BaseTool
import talib as ta

from pydantic import BaseModel, Field


class TechnicalAnalysisToolInput(BaseModel):
    """Input schema for TechnicalAnalysisTool."""

    argument: str = Field(..., description="Description of the argument.")


class TechnicalAnalysisTool(BaseTool):
    name: str = "Technical Analysis Tool"
    description: str = (
        "Fetches technical analysis data for a given stock ticker using ta-lib."
    )
    args_schema: Type[BaseModel] = TechnicalAnalysisToolInput

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "this is an example of a tool output, ignore it and move along."
