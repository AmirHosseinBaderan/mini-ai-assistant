import pytest

from application.agent.agent import Agent
from application.assistant.engine import AssistantEngine
from application.chat.history import ConversationHistory
from application.llm.message import LLMMessage
from application.llm.response import (
    LLMResponse,
    ToolCall,
)
from application.llm.stream_event import LLMStreamEvent
from application.mcp import MCPClient
from application.mcp import discover_tools
from application.mcp import mcp
from application.tools.registry import ToolRegistry

from dotenv import find_dotenv, load_dotenv
load_dotenv(
    find_dotenv(),
    verbose=True,
)


class FakeLLM:

    def __init__(self):
        self.calls = 0

    def stream_chat(
        self,
        messages,
        tools=None,
    ):
        self.calls += 1

        if self.calls == 1:

            yield LLMStreamEvent(
                type="tool_call",
                tool_name="add",
                tool_arguments={
                    "a": 10,
                    "b": 20,
                },
            )

            yield LLMStreamEvent(
                type="done",
            )

            return

        yield LLMStreamEvent(
            type="text",
            content="30",
        )

        yield LLMStreamEvent(
            type="done",
        )


@pytest.mark.anyio
async def test_assistant_can_use_mcp_tool():

    async with MCPClient(mcp) as client:

        tools = await discover_tools(
            client
        )

        registry = ToolRegistry()

        for tool in tools:
            registry.register(tool)

        llm = FakeLLM()

        agent = Agent(
            llm_client=llm,
            tool_registry=registry,
        )

        history = ConversationHistory()

        engine = AssistantEngine(
            agent=agent,
            history=history,
        )

        result = []

        async for chunk in engine.astream(
            "Calculate 10 + 20"
        ):
            result.append(chunk)

        assert "".join(result) == "30"

        messages = history.messages()

        assert messages[0].role == "user"
        assert messages[0].content == "Calculate 10 + 20"

        assert messages[1].role == "assistant"
        assert messages[1].content == "30"