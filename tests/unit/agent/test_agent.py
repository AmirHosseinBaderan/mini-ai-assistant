from typing import Any

import pytest

from application.agent.agent import Agent
from application.llm.message import LLMMessage
from application.llm.response import LLMResponse, ToolCall
from application.tools.base import Tool
from application.tools.registry import ToolRegistry
from application.tools.result import ToolResult


class FakeTool(Tool):

    @property
    def name(self) -> str:
        return "fake_tool"

    @property
    def description(self) -> str:
        return "A fake tool."

    @property
    def parameters(self) -> dict[str, Any]:
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
            **kwargs: Any,
    ) -> ToolResult:
        return ToolResult(
            content={
                "result": f"Tool result for: {kwargs['query']}"
            }
        )


class FakeLLM:

    def __init__(
            self,
            responses: list[LLMResponse],
    ):
        self.responses = responses
        self.calls = []
        self._last_response = None

    def chat(
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

        self._last_response = self.responses.pop(0)
        return self._last_response

    def stream(
            self,
            messages,
    ):
        if self._last_response and self._last_response.content:
            yield self._last_response.content


def test_agent_returns_direct_answer():
    llm = FakeLLM(
        responses=[
            LLMResponse(
                content="Hello!",
            )
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

    assert result == "Hello!"

    assert len(llm.calls) == 1


def test_agent_executes_tool():
    llm = FakeLLM(
        responses=[
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        name="fake_tool",
                        arguments={
                            "query": "Python",
                        },
                    )
                ]
            ),
            LLMResponse(
                content="Python information found.",
            ),
        ]
    )

    registry = ToolRegistry()

    registry.register(
        FakeTool()
    )

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

    assert result == (
        "Python information found."
    )

    assert len(llm.calls) == 2

    second_messages = llm.calls[1]["messages"]

    assert second_messages[-1].role == "tool"

    assert second_messages[-1].content == (
        '{"result": "Tool result for: Python"}'
    )

    assert second_messages[-1].tool_name == (
        "fake_tool"
    )


def test_agent_raises_for_unknown_tool():
    llm = FakeLLM(
        responses=[
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        name="unknown_tool",
                        arguments={},
                    )
                ]
            )
        ]
    )

    registry = ToolRegistry()

    agent = Agent(
        llm_client=llm,
        tool_registry=registry,
    )

    with pytest.raises(KeyError, match="Tool not found"):
        list(
            agent.stream(
                [
                    LLMMessage(
                        role="user",
                        content="Hello",
                    )
                ]
            )
        )


def test_agent_executes_multiple_tool_calls():
    llm = FakeLLM(
        responses=[
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        name="fake_tool",
                        arguments={
                            "query": "Python",
                        },
                    ),
                    ToolCall(
                        name="fake_tool",
                        arguments={
                            "query": "Django",
                        },
                    ),
                ]
            ),
            LLMResponse(
                content="Information found.",
            ),
        ]
    )

    registry = ToolRegistry()

    registry.register(
        FakeTool()
    )

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

    assert result == "Information found."

    assert len(llm.calls) == 2

    second_messages = llm.calls[1]["messages"]

    assert len(second_messages) == 4

    assert second_messages[0].role == "user"

    assert second_messages[1].role == "assistant"

    assert second_messages[2].role == "tool"
    assert second_messages[2].tool_name == "fake_tool"

    assert second_messages[3].role == "tool"
    assert second_messages[3].tool_name == "fake_tool"
