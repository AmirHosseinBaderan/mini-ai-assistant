import json
from collections.abc import Iterator
from typing import Callable

from application.llm.client import LLMClient
from application.llm.message import LLMMessage
from application.tools.registry import ToolRegistry


class Agent:

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        on_tool_call: Callable[[str], None] | None = None,
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

            response = self.llm_client.chat(
                messages=messages,
                tools=self.tool_registry.llm_tools(),
            )

            if not response.has_tool_calls:

                if response.content:
                    yield response.content

                return

            messages.append(
                LLMMessage(
                    role="assistant",
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                )
            )

            for tool_call in response.tool_calls:

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