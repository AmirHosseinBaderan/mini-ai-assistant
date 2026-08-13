import pytest

from application.mcp import MCPClient
from application.mcp.server.bootstrap import create_server


@pytest.mark.anyio
async def test_product_search_mcp():

    mcp = create_server()

    async with MCPClient(
        mcp,
    ) as client:

        tools = await client.list_tools()

        assert any(
            tool.name == "product_search"
            for tool in tools
        )

        result = await client.call_tool(
            name="product_search",
            arguments={
                "query": "iphone 16",
            },
        )

        assert not result.is_error
        assert result.structured_content

        products = result.structured_content["result"]
        print(f'products count: {products}')
        assert products

        for product in products:
            print(f"name : {product['name']}")
            assert product["name"]
            assert product["url"]
            assert product["source"] == "torob"