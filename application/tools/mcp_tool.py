from typing import Any

from application.mcp import MCPClient
from application.tools.result import ToolResult


class MCPTool:

    def __init__(
        self,
        client: MCPClient,
        name: str,
        description: str | None,
        input_schema: dict[str, Any],
    ):
        self.client = client
        self.name = name
        self.description = description or ""
        self.parameters = input_schema

    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

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