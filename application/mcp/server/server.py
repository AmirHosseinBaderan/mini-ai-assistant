from mcp.server import MCPServer

from application.mcp.server.tools.product_search import ProductSearchTool


mcp = MCPServer(
    name="Product Search MCP",
    version="1.0.0",
)


def register_product_search(
    tool: ProductSearchTool,
) -> None:

    async def search_products(
        query: str,
    ) -> list[dict]:

        return await tool.execute(
            query=query,
        )

    mcp.add_tool(
        search_products,
        name="product_search",
        description=(
            "Search for products and prices "
            "from configured shopping websites."
        ),
    )