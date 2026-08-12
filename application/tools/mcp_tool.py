from typing import Any

from application.mcp import MCPClient


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
        self.input_schema = input_schema

    @property
    def parameters(self) -> dict[str, Any]:
        return self.input_schema

    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    async def execute(
        self,
        **kwargs: Any,
    ):
        return await self.client.call_tool(
            self.name,
            kwargs,
        )