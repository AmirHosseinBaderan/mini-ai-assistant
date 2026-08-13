from application.mcp.client.client import MCPClient
from application.tools.mcp_tool import MCPTool


async def discover_tools(
    client: MCPClient,
) -> list[MCPTool]:

    tools = await client.list_tools()

    return [
        MCPTool(
            client=client,
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
        )
        for tool in tools
    ]


async def discover_tools_from_clients(
    clients: list[MCPClient],
) -> list[MCPTool]:

    discovered_tools: list[MCPTool] = []

    for client in clients:
        discovered_tools.extend(
            await discover_tools(client)
        )

    return discovered_tools