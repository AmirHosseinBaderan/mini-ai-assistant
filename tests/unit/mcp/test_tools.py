import pytest

from application.mcp import mcp
from application.mcp import MCPClient
from application.mcp import discover_tools


@pytest.mark.anyio
async def test_discover_mcp_tools():

    async with MCPClient(mcp) as client:

        tools = await discover_tools(
            client
        )

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

        assert add_tool.description

        assert add_tool.input_schema


@pytest.mark.anyio
async def test_mcp_tool_execution():

    async with MCPClient(mcp) as client:

        tools = await discover_tools(
            client
        )

        add_tool = next(
            tool
            for tool in tools
            if tool.name == "add"
        )

        result = await add_tool.execute(
            a=10,
            b=20,
        )

        assert result.structured_content == {
            "result": 30,
        }