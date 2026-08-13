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
            "Save a key-value pair to persistent memory storage. "
            "ALWAYS call this tool whenever the user shares personal "
            "information about themselves (their name, preferences, "
            "language, or any fact they want remembered), even if they "
            "don't explicitly say 'save' or 'remember' — for example if "
            "they say 'my name is X' or 'I prefer dark mode', call this "
            "tool immediately without asking. "
            "Example: save_memory(key='name', value='Amir')."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": (
                        "The key to store the value under. "
                        "For example: 'name', 'language', 'preference'."
                    ),
                },
                "value": {
                    "type": "string",
                    "description": (
                        "The value to store. "
                        "For example: 'Amir', 'Python', 'dark mode'."
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

        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing memory if file exists
        memory_data = {}
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    memory_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                memory_data = {}

        # Update with new key-value pair
        memory_data[key.strip()] = value.strip()

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=2)

        return ToolResult(
            content=f"Successfully saved memory: {key.strip()} = {value.strip()}",
            success=True,
        )
