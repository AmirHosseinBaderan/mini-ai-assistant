from dataclasses import dataclass
from typing import Any, Literal


StreamEventType = Literal[
    "text",
    "tool_call",
    "done",
]


@dataclass
class LLMStreamEvent:
    type: StreamEventType

    content: str | None = None

    tool_name: str | None = None

    tool_arguments: dict[str, Any] | None = None