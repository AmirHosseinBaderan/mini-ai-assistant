from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:

    content: Any

    success: bool = True