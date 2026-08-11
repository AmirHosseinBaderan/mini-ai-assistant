from dataclasses import dataclass
from typing import Any

@dataclass
class LLMMessage:

    role: str
    content: str = ""
    tool_calls: list[Any] | None = None
    tool_name: str | None = None
