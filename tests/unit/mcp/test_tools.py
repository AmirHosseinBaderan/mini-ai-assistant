import pytest

from application.mcp import MCPClient
from application.mcp import discover_tools_from_clients
from application.mcp import mcp


@pytest.mark.anyio
async def test_discover_tools_from_multiple_clients():

    client_a = MCPClient(mcp)
    client_b = MCPClient(mcp)

    async with client_a, client_b:

        tools = await discover_tools_from_clients(
            [
                client_a,
                client_b,
            ]
        )

        names = [
            tool.name
            for tool in tools
        ]

        assert names.count("add") == 2