from typing import Any

from application.agent.agent import Agent
from application.llm.response import LLMResponse, ToolCall
from application.tools.base import Tool
from application.tools.registry import ToolRegistry


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
    ) -> Any:
        return {
            "result": f"Tool result for: {kwargs['query']}"
        }


class FakeLLM:


    def __init__(
            self,
            responses: list[LLMResponse],
    ):
        self.responses = responses
        self.calls = []


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

        return self.responses.pop(0)


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

    result = agent.run("Hello")

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

    result = agent.run(
        "Tell me about Python."
    )

    assert result == "Python information found."

    assert len(llm.calls) == 2

    second_messages = llm.calls[1]["messages"]

    assert second_messages[-1] == {
        "role": "tool",
        "content": (
            "{'result': "
            "'Tool result for: Python'}"
        ),
        "tool_name": "fake_tool",
    }


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

    try:
        agent.run("Use the unknown tool.")
        assert False
    except KeyError as error:
        assert "Tool not found" in str(error)

