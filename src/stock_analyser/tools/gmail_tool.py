import os
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from stock_analyser.tools.gmail_utility_inline_images import authenticate_gmail, create_message_with_inline_image, create_draft

from dotenv import load_dotenv
load_dotenv()

class GmailToolInput(BaseModel):
    """Input schema for MyCustomTool."""

    body: str = Field(..., description="The body of the email to send.")
    subject: str = Field(..., description="The subject of the email to send.")


class GmailTool(BaseTool):
    name: str = "GmailTool"
    description: str = (
        "Send an email using the provided subject and body"
    )
    args_schema: Type[BaseModel] = GmailToolInput

    def _run(self, body: str, subject: str) -> str:
        try:
            service = authenticate_gmail()
            sender = os.getenv("SENDER_EMAIL")   
            to = os.getenv("CLIENT_EMAIL")
            subject = subject
            message_text = create_message_with_inline_image(sender, to, subject, body)
            draft = create_draft(service, "me", message_text)
            return f"Email draft created successfully! Draft ID: {draft['id']}"
        except Exception as e:
            return f"An error occurred: {e}"
        
if __name__ == "__main__":
    gmail_tool = GmailTool()

    report = "/home/j/ai/crewAI/finance/stock_analyser/final_reports/2025-03-12/MSFT_Stock_Analysis_Report_20250312_132728.md"

    with open(report, "r") as file:
        body = file.read()

    print(gmail_tool.run(body=body, subject="Test Email"))
        