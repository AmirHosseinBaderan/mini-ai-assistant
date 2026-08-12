from typing import Any

from mcp import Client


class MCPClient:

    def __init__(
        self,
        server,
    ):
        self._server = server
        self._client = Client(server)
        self._session = None

    async def __aenter__(self):

        self._session = await self._client.__aenter__()

        return self

    async def __aexit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        await self._client.__aexit__(
            exc_type,
            exc_value,
            traceback,
        )

        self._session = None

    async def list_tools(self):

        if self._session is None:
            raise RuntimeError(
                "MCP client is not connected"
            )

        result = await self._session.list_tools()

        return result.tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ):

        if self._session is None:
            raise RuntimeError(
                "MCP client is not connected"
            )

        return await self._session.call_tool(
            name,
            arguments,
        )