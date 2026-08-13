from application.mcp.client.client import MCPClient


class MCPToolProvider:

    def __init__(
        self,
        client: MCPClient,
    ):
        self.client = client

    async def tools(self):

        return await self.client.list_tools()