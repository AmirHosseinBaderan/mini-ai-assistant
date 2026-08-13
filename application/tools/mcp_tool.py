from typing import Any

from application.mcp.client.client import MCPClient
from application.tools.base import Tool
from application.tools.result import ToolResult


class MCPTool(Tool):

    def __init__(
        self,
        client: MCPClient,
        name: str,
        description: str | None,
        input_schema: dict[str, Any],
    ):
        self.client = client
        self._name = name
        self._description = description or ""
        self._parameters = input_schema

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(
        self,
        **kwargs: Any,
    ) -> ToolResult:

        result = await self.client.call_tool(
            self.name,
            kwargs,
        )

        content = []

        for item in result.content:

            if hasattr(item, "text"):
                content.append(item.text)

            elif hasattr(item, "data"):
                content.append(item.data)

            else:
                content.append(str(item))

        return ToolResult(
            content={
                "result": "\n".join(content),
            },
            success=not result.is_error,
        )