import pytest

from mcp import Client

from application.mcp import mcp


@pytest.mark.anyio
async def test_add_tool():

    async with Client(mcp) as client:

        result = await client.call_tool(
            "add",
            {
                "a": 2,
                "b": 3,
            },
        )

        assert result.structured_content == {
            "result": 5,
        }


@pytest.mark.anyio
async def test_tools_discovery():

    async with Client(mcp) as client:

        result = await client.list_tools()

        tools = result.tools

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

        assert add_tool.name == "add"

        assert add_tool.description

        assert add_tool.input_schema is not None