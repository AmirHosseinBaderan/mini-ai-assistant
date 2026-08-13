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
        "ALWAYS call this tool at the start of a conversation, or whenever "
        "the user refers to something they told you before (e.g. 'what's "
        "my name?', 'do you remember my preference?', 'what did I tell you "
        "earlier?'), to check if relevant information was already saved — "
        "do not rely on chat history alone, since saved memory persists "
        "across sessions and chat history may not. "
        "You can load all memory by omitting the key, or load a specific "
        "value by providing its key. "
        "Example: load_memory(key='name') returns 'Amir'. "
        "Example: load_memory() returns all saved key-value pairs. "
        "If the requested key does not exist, this returns an empty result "
        "rather than an error — treat that as 'nothing saved yet', not a failure."
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
