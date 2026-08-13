import pytest

from application.mcp.server.server import mcp


@pytest.mark.anyio
async def test_product_search_tool_discovery():

    result = await mcp.list_tools()

    tools = {
        tool.name: tool
        for tool in result
    }

    assert "search_products" in tools

    tool = tools["search_products"]

    assert tool.description
    assert tool.input_schema