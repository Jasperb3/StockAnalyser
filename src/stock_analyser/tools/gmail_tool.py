import os
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from .gmail_utility_inline_images import authenticate_gmail, create_message_with_inline_image, create_draft

from dotenv import load_dotenv
load_dotenv()

class GmailToolInput(BaseModel):
    """Input schema for MyCustomTool."""

    body: str = Field(..., description="The body of the email to send.")
    subject: str = Field(..., description="The subject of the email to send.")


class GmailTool(BaseTool):
    name: str = "GmailTool"
    description: str = (
        "Send an email to Zarina with the news report using the provided subject '{subject}' and body: {body}"
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
        