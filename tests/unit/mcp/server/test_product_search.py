import pytest

from application.mcp.server.tools.product_search import ProductSearchTool
from application.mcp.server.server import (
    mcp,
    register_product_search,
)
from application.product_search.models import Product


class FakeEngine:

    def __init__(self):

        self.calls = []

    async def search(
        self,
        query: str,
    ) -> list[Product]:

        self.calls.append(
            query,
        )

        return [
            Product(
                name="iPhone 16",
                price="160000000",
                url="https://torob.com/product/123",
                source="torob",
            ),
            Product(
                name="iPhone 16 Pro",
                price="200000000",
                url="https://torob.com/product/456",
                source="torob",
            ),
        ]


@pytest.mark.anyio
async def test_product_search():

    engine = FakeEngine()

    tool = ProductSearchTool(
        engine=engine,
    )

    register_product_search(
        tool,
    )

    tools = await mcp.list_tools()

    assert any(
        item.name == "product_search"
        for item in tools
    )

    result = await mcp.call_tool(
        "product_search",
        {
            "query": "iphone 16",
        },
    )

    assert engine.calls == [
        "iphone 16",
    ]

    assert result.structured_content == {
        "result": [
            {
                "name": "iPhone 16",
                "price": "160000000",
                "url": "https://torob.com/product/123",
                "source": "torob",
            },
            {
                "name": "iPhone 16 Pro",
                "price": "200000000",
                "url": "https://torob.com/product/456",
                "source": "torob",
            },
        ],
    }