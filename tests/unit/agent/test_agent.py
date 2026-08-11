from unittest.mock import Mock

import pytest

from application.agent.agent import Agent
from application.llm.message import LLMMessage
from application.llm.stream_event import LLMStreamEvent
from application.llm.response import ToolCall
from application.tools.base import Tool
from application.tools.registry import ToolRegistry


class FakeLLM:

    def __init__(
        self,
        responses: list[list[LLMStreamEvent]],
    ):
        self.responses = responses
        self.call_count = 0
        self.calls = []

    def stream_chat(
        self,
        messages,
        tools,
    ):
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
            }
        )

        response = self.responses[
            self.call_count
        ]

        self.call_count += 1

        yield from response


class FakeTool(Tool):

    def __init__(self):
        self.execute_calls = []

    @property
    def name(self) -> str:
        return "fake_tool"

    @property
    def description(self) -> str:
        return "A fake tool for testing."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                },
            },
            "required": ["query"],
        }

    def execute(
        self,
        **kwargs,
    ):
        self.execute_calls.append(kwargs)

        return Mock(
            content={
                "result": (
                    f"Tool result for: "
                    f"{kwargs['query']}"
                )
            },
            success=True,
        )


def test_agent_streams_direct_answer():

    llm = FakeLLM(
        responses=[
            [
                LLMStreamEvent(
                    type="text",
                    content="Hello",
                ),
                LLMStreamEvent(
                    type="text",
                    content=" Amir",
                ),
                LLMStreamEvent(
                    type="done",
                ),
            ]
        ]
    )

    registry = ToolRegistry()

    agent = Agent(
        llm_client=llm,
        tool_registry=registry,
    )

    result = "".join(
        agent.stream(
            [
                LLMMessage(
                    role="user",
                    content="Hello",
                )
            ]
        )
    )

    assert result == "Hello Amir"

    assert llm.call_count == 1


def test_agent_executes_tool():

    llm = FakeLLM(
        responses=[
            [
                LLMStreamEvent(
                    type="text",
                    content="Let me search.",
                ),
                LLMStreamEvent(
                    type="tool_call",
                    tool_name="fake_tool",
                    tool_arguments={
                        "query": "Python",
                    },
                ),
                LLMStreamEvent(
                    type="done",
                ),
            ],
            [
                LLMStreamEvent(
                    type="text",
                    content="Python information found.",
                ),
                LLMStreamEvent(
                    type="done",
                ),
            ],
        ]
    )

    tool = FakeTool()

    registry = ToolRegistry()

    registry.register(tool)

    agent = Agent(
        llm_client=llm,
        tool_registry=registry,
    )

    result = "".join(
        agent.stream(
            [
                LLMMessage(
                    role="user",
                    content=(
                        "Tell me about Python."
                    ),
                )
            ]
        )
    )

    assert result == (
        "Let me search."
        "Python information found."
    )

    assert llm.call_count == 2

    assert tool.execute_calls == [
        {
            "query": "Python",
        }
    ]


def test_agent_raises_for_unknown_tool():

    llm = FakeLLM(
        responses=[
            [
                LLMStreamEvent(
                    type="tool_call",
                    tool_name="unknown_tool",
                    tool_arguments={},
                ),
                LLMStreamEvent(
                    type="done",
                ),
            ],
        ]
    )

    registry = ToolRegistry()

    agent = Agent(
        llm_client=llm,
        tool_registry=registry,
    )

    with pytest.raises(
        KeyError,
        match="Tool not found",
    ):
        list(
            agent.stream(
                [
                    LLMMessage(
                        role="user",
                        content=(
                            "Use the unknown tool."
                        ),
                    )
                ]
            )
        )


def test_agent_executes_multiple_tool_calls():

    llm = FakeLLM(
        responses=[
            [
                LLMStreamEvent(
                    type="tool_call",
                    tool_name="fake_tool",
                    tool_arguments={
                        "query": "Python",
                    },
                ),
                LLMStreamEvent(
                    type="tool_call",
                    tool_name="fake_tool",
                    tool_arguments={
                        "query": "Django",
                    },
                ),
                LLMStreamEvent(
                    type="done",
                ),
            ],
            [
                LLMStreamEvent(
                    type="text",
                    content="Information found.",
                ),
                LLMStreamEvent(
                    type="done",
                ),
            ],
        ]
    )

    tool = FakeTool()

    registry = ToolRegistry()

    registry.register(tool)

    agent = Agent(
        llm_client=llm,
        tool_registry=registry,
    )

    result = "".join(
        agent.stream(
            [
                LLMMessage(
                    role="user",
                    content=(
                        "Tell me about "
                        "Python and Django."
                    ),
                )
            ]
        )
    )

    assert result == "Information found."

    assert llm.call_count == 2

    assert tool.execute_calls == [
        {
            "query": "Python",
        },
        {
            "query": "Django",
        },
    ]


def test_agent_calls_on_tool_call_callback():

    llm = FakeLLM(
        responses=[
            [
                LLMStreamEvent(
                    type="tool_call",
                    tool_name="fake_tool",
                    tool_arguments={
                        "query": "Python",
                    },
                ),
                LLMStreamEvent(
                    type="done",
                ),
            ],
            [
                LLMStreamEvent(
                    type="text",
                    content="Done.",
                ),
                LLMStreamEvent(
                    type="done",
                ),
            ],
        ]
    )

    tool = FakeTool()

    registry = ToolRegistry()

    registry.register(tool)

    on_tool_call = Mock()

    agent = Agent(
        llm_client=llm,
        tool_registry=registry,
        on_tool_call=on_tool_call,
    )

    result = "".join(
        agent.stream(
            [
                LLMMessage(
                    role="user",
                    content="Search Python.",
                )
            ]
        )
    )

    assert result == "Done."

    on_tool_call.assert_called_once_with(
        "fake_tool"
    )


def test_agent_preserves_streamed_text_before_tool():

    llm = FakeLLM(
        responses=[
            [
                LLMStreamEvent(
                    type="text",
                    content="I will ",
                ),
                LLMStreamEvent(
                    type="text",
                    content="search first.",
                ),
                LLMStreamEvent(
                    type="tool_call",
                    tool_name="fake_tool",
                    tool_arguments={
                        "query": "Python",
                    },
                ),
                LLMStreamEvent(
                    type="done",
                ),
            ],
            [
                LLMStreamEvent(
                    type="text",
                    content="Here is the result.",
                ),
                LLMStreamEvent(
                    type="done",
                ),
            ],
        ]
    )

    tool = FakeTool()

    registry = ToolRegistry()

    registry.register(tool)

    agent = Agent(
        llm_client=llm,
        tool_registry=registry,
    )

    result = "".join(
        agent.stream(
            [
                LLMMessage(
                    role="user",
                    content="Search Python.",
                )
            ]
        )
    )

    assert result == (
        "I will "
        "search first."
        "Here is the result."
    )