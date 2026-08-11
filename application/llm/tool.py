from dataclasses import dataclass
from typing import Any

@dataclass
class LLMTool:

    name: str
    description: str
    parameters: dict[str, Any]