from mcp.server import MCPServer

from application.mcp import search_products


mcp = MCPServer(
    name="Product Search MCP",
    version="1.0.0",
)


mcp.add_tool(search_products)