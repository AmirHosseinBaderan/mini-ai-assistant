import pytest

from application.agent.agent import Agent
from application.llm.message import LLMMessage
from application.llm.stream_event import LLMStreamEvent
from application.tools.registry import ToolRegistry


class FakeAsyncTool:

    name = "add"
    description = "Add two numbers."

    parameters = {
        "type": "object",
        "properties": {
            "a": {
                "type": "integer",
            },
            "b": {
                "type": "integer",
            },
        },
        "required": [
            "a",
            "b",
        ],
    }

    async def execute(self, **kwargs):

        return type(
            "ToolResult",
            (),
            {
                "content": {
                    "result": kwargs["a"] + kwargs["b"]
                }
            },
        )()


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
async def test_agent_astream_executes_async_tool():

    registry = ToolRegistry()

    registry.register(
        FakeAsyncTool()
    )

    agent = Agent(
        llm_client=FakeLLM(),
        tool_registry=registry,
    )

    result = []

    async for token in agent.astream(
        [
            LLMMessage(
                role="user",
                content="Calculate 10 + 20",
            )
        ]
    ):
        result.append(token)

    assert "".join(result) == "30"