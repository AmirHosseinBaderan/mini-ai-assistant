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
            "Load saved user memory from persistent storage. "
            "Use this tool when you need information saved from previous "
            "sessions, such as the user's name, preferences, projects, "
            "or other persistent context. "
            "Provide a key to load one memory value, or omit the key "
            "to load all saved memory."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": (
                        "Optional key to load a specific value. "
                        "If omitted, all saved memory is returned."
                    ),
                },
            },
            "required": [],
        }

    def execute(
        self,
        **kwargs: Any,
    ) -> ToolResult:
        key = kwargs.get("key")

        if not self.path.exists():
            return ToolResult(
                content="No memory found. Use save_memory to store details first.",
                success=True,
            )

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                memory_data = json.load(f)

            if not memory_data:
                return ToolResult(
                    content="Memory file exists but is empty.",
                    success=True,
                )

            # If key is provided, return the specific value
            if key is not None:
                key = key.strip()
                if key not in memory_data:
                    return ToolResult(
                        content=f"Key '{key}' not found in memory.",
                        success=True,
                    )
                return ToolResult(
                    content=memory_data[key],
                    success=True,
                )

            # Otherwise return all memory
            return ToolResult(
                content=memory_data,
                success=True,
            )

        except (json.JSONDecodeError, IOError) as e:
            return ToolResult(
                content=f"Error loading memory: {str(e)}",
                success=False,
            )
