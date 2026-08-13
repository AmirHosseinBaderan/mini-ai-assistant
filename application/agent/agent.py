import json
from collections.abc import AsyncIterator, Iterator

from application.llm.client import LLMClient
from application.llm.message import LLMMessage
from application.llm.response import ToolCall
from application.tools.registry import ToolRegistry


class Agent:

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        on_tool_call=None,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.on_tool_call = on_tool_call

    def stream(
        self,
        messages: list[LLMMessage],
    ) -> Iterator[str]:

        messages = messages.copy()

        while True:

            tool_calls = []

            for event in self.llm_client.stream_chat(
                messages=messages,
                tools=self.tool_registry.llm_tools(),
            ):

                if event.type == "text":

                    if event.content:
                        yield event.content

                    continue

                if event.type == "tool_call":

                    tool_calls.append(
                        ToolCall(
                            name=event.tool_name,
                            arguments=(
                                event.tool_arguments
                                or {}
                            ),
                        )
                    )

                    continue

                if event.type == "done":
                    break

            if not tool_calls:
                return

            messages.append(
                LLMMessage(
                    role="assistant",
                    content="",
                    tool_calls=tool_calls,
                )
            )

            for tool_call in tool_calls:

                if self.on_tool_call:
                    self.on_tool_call(
                        tool_call.name
                    )

                tool = self.tool_registry.get(
                    tool_call.name
                )

                result = tool.execute(
                    **tool_call.arguments
                )

                messages.append(
                    LLMMessage(
                        role="tool",
                        content=json.dumps(
                            result.content,
                            ensure_ascii=False,
                        ),
                        tool_name=tool_call.name,
                    )
                )

    async def astream(
            self,
            messages: list[LLMMessage],
    ) -> AsyncIterator[str]:

        messages = messages.copy()

        while True:

            tool_calls = []

            for event in self.llm_client.stream_chat(
                    messages=messages,
                    tools=self.tool_registry.llm_tools(),
            ):

                if event.type == "text":

                    if event.content:
                        yield event.content

                    continue

                if event.type == "tool_call":
                    tool_calls.append(
                        ToolCall(
                            name=event.tool_name,
                            arguments=(
                                    event.tool_arguments
                                    or {}
                            ),
                        )
                    )

                    continue

                if event.type == "done":
                    break

            if not tool_calls:
                return

            messages.append(
                LLMMessage(
                    role="assistant",
                    content="",
                    tool_calls=tool_calls,
                )
            )

            for tool_call in tool_calls:

                if self.on_tool_call:
                    self.on_tool_call(
                        tool_call.name
                    )

                tool = self.tool_registry.get(
                    tool_call.name
                )

                result = await tool.execute(
                    **tool_call.arguments
                )

                messages.append(
                    LLMMessage(
                        role="tool",
                        content=json.dumps(
                            result.content,
                            ensure_ascii=False,
                        ),
                        tool_name=tool_call.name,
                    )
                )