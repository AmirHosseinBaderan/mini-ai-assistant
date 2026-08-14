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
        take: int = 10,
    ) -> list[dict]:

        return await tool.execute(
            query=query,
            take=take,
        )

    mcp.add_tool(
        search_products,
        name="product_search",
        description=(
            "Search configured shopping websites for products and prices. "
            "Returns up to 'take' matching products (default 10), each including "
            "the product name, price, URL, and source website. "
            "Use this tool to find products, compare prices, or check "
            "availability across different stores."
        ),
    )