import pytest

from application.mcp import mcp
from application.mcp import MCPClient


@pytest.mark.anyio
async def test_mcp_client_discovers_tools():

    async with MCPClient(mcp) as client:

        tools = await client.list_tools()

        assert len(tools) > 0

        add_tool = next(
            (
                tool
                for tool in tools
                if tool.name == "add"
            ),
            None,
        )

        assert add_tool is not None


@pytest.mark.anyio
async def test_mcp_client_calls_tool():

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