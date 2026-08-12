from sympy import true

from application.tools.base import Tool
from typing import Any

from application.tools.result import ToolResult


class SaveMemoryTool(Tool):

    @property
    def name(self) -> str:
        return "save_memory"

    @property
    def description(self) -> str:
        return (
            "Save details in memory"
            "for save user , chat , and important details in memory"
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