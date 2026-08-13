from application.mcp.client.client import MCPClient
from application.mcp.tools import discover_tools_from_clients
from application.tools.registry import ToolRegistry


async def register_mcp_tools(
    registry: ToolRegistry,
    clients: list[MCPClient],
) -> None:

    tools = await discover_tools_from_clients(
        clients,
    )

    for tool in tools:

        registry.register(
            tool,
        )