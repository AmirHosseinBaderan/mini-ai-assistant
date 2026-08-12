import json
from pathlib import Path
from typing import Any

from application.tools.base import Tool
from application.tools.result import ToolResult


class UpdateMemoryTool(Tool):

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
        return "update_memory"

    @property
    def description(self) -> str:
        return (
            "Update an existing key-value pair in persistent memory storage. "
            "Use this to modify previously saved user preferences, personal information, "
            "or any details that were saved using save_memory. "
            "Example: update_memory(key='name', value='Ali'). "
            "Returns an error if the key does not exist."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": (
                        "The key to update. Must already exist in memory. "
                        "For example: 'name', 'language', 'preference'."
                    ),
                },
                "value": {
                    "type": "string",
                    "description": (
                        "The new value to set. "
                        "For example: 'Ali', 'JavaScript', 'light mode'."
                    ),
                },
            },
            "required": ["key", "value"],
        }

    def execute(
        self,
        **kwargs: Any,
    ) -> ToolResult:
        key = kwargs.get("key")
        value = kwargs.get("value")

        if not isinstance(key, str) or not key.strip():
            raise ValueError(
                "key is required and must be a non-empty string"
            )

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "value is required and must be a non-empty string"
            )

        key = key.strip()
        value = value.strip()

        # Check if memory file exists
        if not self.path.exists():
            return ToolResult(
                content=f"No memory found. Cannot update key '{key}' because memory is empty. Use save_memory first.",
                success=False,
            )

        # Load existing memory
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                memory_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            return ToolResult(
                content=f"Error reading memory: {str(e)}",
                success=False,
            )

        # Check if key exists
        if key not in memory_data:
            return ToolResult(
                content=f"Key '{key}' not found in memory. Use save_memory to create it first.",
                success=False,
            )

        # Update the key-value pair
        old_value = memory_data[key]
        memory_data[key] = value

        # Save back to file
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(memory_data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            return ToolResult(
                content=f"Error saving memory: {str(e)}",
                success=False,
            )

        return ToolResult(
            content=f"Successfully updated memory: {key} changed from '{old_value}' to '{value}'",
            success=True,
        )
