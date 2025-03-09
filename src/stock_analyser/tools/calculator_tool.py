from crewai.tools import BaseTool


class CalculatorTool(BaseTool):
    name: str = "Calculator tool"
    description: str = (
        "A tool for evaluating mathematical expressions including addition, subtraction, multiplication and division. Input should be a valid mathematical expression like `200*7` or `5000/2*10`."
    )

    def _run(self, operation: str) -> int:
        # Implementation goes here
        return eval(operation)