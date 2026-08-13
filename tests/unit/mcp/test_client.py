import pytest

from application.mcp import MCPClient
from application.mcp import mcp


@pytest.mark.anyio
async def test_client_cannot_list_tools_before_connect():

    client = MCPClient(mcp)

    with pytest.raises(
        RuntimeError,
        match="MCP client is not connected",
    ):
        await client.list_tools()


@pytest.mark.anyio
async def test_client_cannot_call_tool_before_connect():

    client = MCPClient(mcp)

    with pytest.raises(
        RuntimeError,
        match="MCP client is not connected",
    ):
        await client.call_tool(
            "add",
            {
                "a": 1,
                "b": 2,
            },
        )


@pytest.mark.anyio
async def test_client_can_list_tools_after_connect():

    async with MCPClient(mcp) as client:

        tools = await client.list_tools()

        names = [
            tool.name
            for tool in tools
        ]

        assert "add" in names


@pytest.mark.anyio
async def test_client_can_call_tool_after_connect():

    async with MCPClient(mcp) as client:

        result = await client.call_tool(
            "add",
            {
                "a": 10,
                "b": 20,
            },
        )

        assert result.structured_content == {
            "result": 30,
        }


@pytest.mark.anyio
async def test_client_is_disconnected_after_context_exit():

    client = MCPClient(mcp)

    async with client:

        assert client._session is not None

    assert client._session is None


@pytest.mark.anyio
async def test_client_cannot_use_tools_after_context_exit():

    client = MCPClient(mcp)

    async with client:
        pass

    with pytest.raises(
        RuntimeError,
        match="MCP client is not connected",
    ):
        await client.list_tools()