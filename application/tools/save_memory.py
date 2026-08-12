import json
from pathlib import Path
from typing import Any

from application.tools.base import Tool
from application.tools.result import ToolResult


class SaveMemoryTool(Tool):

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
        return "save_memory"

    @property
    def description(self) -> str:
        return (
            "Save important details to persistent memory storage. "
            "Use this to remember user preferences, personal information, "
            "chat context, or any details that should persist across sessions. "
            "The saved memory can be retrieved later using load_memory."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "details": {
                    "type": "string",
                    "description": (
                        "The details to save in memory. "
                        "Can be any text, facts, user preferences, "
                        "or important information to remember."
                    ),
                },
            },
            "required": ["details"],
        }

    def execute(
            self,
            **kwargs: Any,
    ) -> ToolResult:
        details = kwargs.get("details")

        if not isinstance(details, str) or not details.strip():
            raise ValueError(
                "details is required and must be a non-empty string"
            )

        memory_data = {"memory": details.strip()}

        self.path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=2)

        return ToolResult(
            content=f"Successfully saved memory: {details.strip()}",
            success=True,
        )
