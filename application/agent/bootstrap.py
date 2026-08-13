from application.agent.agent import Agent
from application.llm.client import LLMClient
from application.mcp.client.client import MCPClient
from application.tools.bootstrap import register_mcp_tools
from application.tools.registry import ToolRegistry


async def create_agent(
    llm_client: LLMClient,
    mcp_clients: list[MCPClient],
) -> Agent:

    registry = ToolRegistry()

    await register_mcp_tools(
        registry=registry,
        clients=mcp_clients,
    )

    return Agent(
        llm_client=llm_client,
        tool_registry=registry,
    )