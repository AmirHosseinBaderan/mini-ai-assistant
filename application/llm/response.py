from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)