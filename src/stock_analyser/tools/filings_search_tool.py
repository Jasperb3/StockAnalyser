from typing import Any, Optional, Type
from pathlib import Path
from embedchain.loaders.directory_loader import DirectoryLoader
from pydantic import BaseModel, Field

from crewai_tools.tools.rag.rag_tool import RagTool

FILINGS_DIR = Path(__file__).parent.parent.parent.parent / "filings"

class FixedDirectorySearchToolSchema(BaseModel):
    """Input for DirectorySearchTool."""

    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search the directory's content",
    )


class DirectorySearchToolSchema(FixedDirectorySearchToolSchema):
    """Input for DirectorySearchTool."""

    directory: str = Field(..., description="Mandatory directory you want to search")


class FilingsSearchTool(RagTool):
    name: str = "Search within SEC filings"
    description: str = (
        "A tool that can be used to semantic search a query in the SEC 10-K, 10-Q, 8-K, and 6-K (if available) filings content."
    )
    args_schema: Type[BaseModel] = DirectorySearchToolSchema

    def __init__(self, directory: Optional[str] = str(FILINGS_DIR), **kwargs):
        super().__init__(**kwargs)
        if directory is not None:
            kwargs["loader"] = DirectoryLoader(config=dict(recursive=True))
            self.add(directory)
            self.description = f"A tool that can be used to semantic search a query in the SEC 10-K, 10-Q, 8-K, and 6-K (if available) filings content."
            self.args_schema = FixedDirectorySearchToolSchema
            self._generate_description()

    def add(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().add(*args, **kwargs)

    def _before_run(
        self,
        query: str,
        **kwargs: Any,
    ) -> Any:
        if "directory" in kwargs:
            self.add(kwargs["directory"])

    def _run(
        self,
        search_query: str,
        **kwargs: Any,
    ) -> Any:
        return super()._run(query=search_query, **kwargs)
