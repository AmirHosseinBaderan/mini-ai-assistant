import json
from pathlib import Path
from typing import Any

from application.tools.base import Tool
from application.tools.result import ToolResult


class LoadMemoryTool(Tool):

    def __init__(
            self,
            path: str | None = None,
    ):
        if path is None:
            self.path = Path("data/memory") / "memo.json"
        else:
            self.path = Path(path)

    @property
    def name(self) -> str:
        return "load_memory"

    @property
    def description(self) -> str:
        return (
            "Load previously saved memory details from persistent storage. "
            "Use this to retrieve user preferences, personal information, "
            "chat context, or any details that were saved using save_memory. "
            "Returns the stored memory content if available."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def execute(
            self,
            **kwargs: Any,
    ) -> ToolResult:
        if not self.path.exists():
            return ToolResult(
                content="No memory found. Use save_memory to store details first.",
                success=True,
            )

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                memory_data = json.load(f)

            memory_content = memory_data.get("memory", "")

            if not memory_content:
                return ToolResult(
                    content="Memory file exists but is empty.",
                    success=True,
                )

            return ToolResult(
                content=memory_content,
                success=True,
            )

        except (json.JSONDecodeError, IOError) as e:
            return ToolResult(
                content=f"Error loading memory: {str(e)}",
                success=False,
            )
