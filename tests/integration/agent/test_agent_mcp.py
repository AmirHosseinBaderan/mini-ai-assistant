import pytest

from application.agent.agent import Agent
from application.llm.message import LLMMessage
from application.llm.stream_event import LLMStreamEvent
from application.mcp.client.client import MCPClient
from application.mcp.server.bootstrap import create_server
from application.tools.mcp_tool import MCPTool
from application.tools.registry import ToolRegistry


class FakeLLMClient:

    def __init__(self):
        self.calls = 0

    def stream_chat(
        self,
        messages,
        tools,
    ):

        self.calls += 1

        if self.calls == 1:

            yield LLMStreamEvent(
                type="tool_call",
                tool_name="product_search",
                tool_arguments={
                    "query": "iphone 16",
                },
            )

            yield LLMStreamEvent(
                type="done",
            )

            return

        yield LLMStreamEvent(
            type="text",
            content="نتایج جستجوی آیفون ۱۶ دریافت شد.",
        )

        yield LLMStreamEvent(
            type="done",
        )


@pytest.mark.anyio
async def test_agent_mcp_product_search():

    server = create_server()

    async with MCPClient(server) as client:

        tools = await client.list_tools()

        product_tool = next(
            tool
            for tool in tools
            if tool.name == "product_search"
        )

        mcp_tool = MCPTool(
            client=client,
            name=product_tool.name,
            description=product_tool.description,
            input_schema=product_tool.input_schema,
        )

        registry = ToolRegistry()

        registry.register(
            mcp_tool,
        )

        agent = Agent(
            llm_client=FakeLLMClient(),
            tool_registry=registry,
        )

        messages = [
            LLMMessage(
                role="user",
                content="قیمت آیفون ۱۶ چنده؟",
            )
        ]

        result = []

        async for chunk in agent.astream(
            messages,
        ):
            result.append(chunk)

        assert "".join(result) == (
            "نتایج جستجوی آیفون ۱۶ دریافت شد."
        )