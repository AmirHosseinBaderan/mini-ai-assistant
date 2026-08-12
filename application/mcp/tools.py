from application.mcp import MCPClient
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