from collections.abc import Iterator

import pytest

from application.agent.agent import Agent
from application.llm.client import LLMClient
from application.llm.message import LLMMessage
from application.llm.response import LLMResponse
from application.llm.stream_event import LLMStreamEvent
from application.llm.tool import LLMTool
from application.mcp.client.client import MCPClient
from application.mcp.server.server import MCPServer
from application.mcp.tools import discover_tools
from application.tools.registry import ToolRegistry


class FakeLLMClient(LLMClient):

    def __init__(self):
        self.calls = 0

    def chat(
        self,
        messages: list[LLMMessage],
        tools: list[LLMTool],
    ) -> LLMResponse:
        raise NotImplementedError

    def stream(
        self,
        messages: list[LLMMessage],
    ) -> Iterator[str]:
        raise NotImplementedError

    def stream_chat(
        self,
        messages: list[LLMMessage],
        tools: list[LLMTool],
    ) -> Iterator[LLMStreamEvent]:

        self.calls += 1

        if self.calls == 1:

            yield LLMStreamEvent(
                type="tool_call",
                tool_name="echo",
                tool_arguments={
                    "text": "hello",
                },
            )

        else:

            yield LLMStreamEvent(
                type="text",
                content="MCP tool worked.",
            )

        yield LLMStreamEvent(
            type="done",
        )


@pytest.mark.anyio
async def test_agent_mcp_e2e():

    server = MCPServer(
        name="Test MCP",
        version="1.0.0",
    )

    async def echo(
        text: str,
    ) -> str:

        return text

    server.add_tool(
        echo,
        name="echo",
        description="Echo text.",
    )

    llm_client = FakeLLMClient()

    async with MCPClient(server) as mcp_client:

        tools = await discover_tools(
            mcp_client,
        )

        registry = ToolRegistry()

        for tool in tools:
            registry.register(tool)

        agent = Agent(
            llm_client=llm_client,
            tool_registry=registry,
        )

        messages = [
            LLMMessage(
                role="user",
                content="Say hello.",
            )
        ]

        response = []

        async for chunk in agent.astream(
            messages,
        ):
            response.append(chunk)

        assert "".join(response) == (
            "MCP tool worked."
        )

        assert llm_client.calls == 2