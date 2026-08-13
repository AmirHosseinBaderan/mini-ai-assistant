import pytest

from application.agent.agent import Agent
from application.llm.message import LLMMessage
from application.llm.stream_event import LLMStreamEvent
from application.llm.tool import LLMTool
from application.tools.base import Tool
from application.tools.registry import ToolRegistry
from application.tools.result import ToolResult


class FakeTool(Tool):

    @property
    def name(self) -> str:
        return "product_search"

    @property
    def description(self) -> str:
        return "Search products."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                },
            },
            "required": [
                "query",
            ],
        }

    async def execute(
        self,
        **kwargs,
    ) -> ToolResult:

        return ToolResult(
            content={
                "result": (
                    "iPhone 16 - "
                    "160000000 تومان"
                ),
            },
        )


class FakeLLMClient:

    def __init__(self):

        self.calls = 0

    def stream_chat(
        self,
        messages,
        tools: list[LLMTool],
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
            content="قیمت آیفون ۱۶ حدود ۱۶۰ میلیون تومان است.",
        )

        yield LLMStreamEvent(
            type="done",
        )


@pytest.mark.anyio
async def test_agent_mcp_tool_call():

    llm_client = FakeLLMClient()

    registry = ToolRegistry()

    registry.register(
        FakeTool(),
    )

    agent = Agent(
        llm_client=llm_client,
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
        "قیمت آیفون ۱۶ حدود ۱۶۰ میلیون تومان است."
    )

    assert llm_client.calls == 2