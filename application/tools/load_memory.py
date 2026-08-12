from sympy import true

from application.tools.base import Tool
from typing import Any

from application.tools.result import ToolResult


class LoadMemoryTool(Tool):

    @property
    def name(self) -> str:
        return "load_memory"

    @property
    def description(self) -> str:
        return (
            "Load details from memory"
            "for load user , chat , and important details from memory"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {

        }

    def execute(
        self,
        **kwargs: Any,
    ) -> ToolResult:
        return ToolResult(
            "",True
        )