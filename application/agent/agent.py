from typing import Any

from application.llm.client import LLMClient
from application.tools.registry import ToolRegistry

class Agent:

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry

    def run(
        self,
        query: str,
    ) -> str:

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": query,
            }
        ]

        while True:

            response = self.llm_client.chat(
                messages=messages,
                tools=self.tool_registry.schemas(),
            )

            if not response.has_tool_calls:
                return response.content or ""

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": response.tool_calls,
                }
            )

            for tool_call in response.tool_calls:

                tool = self.tool_registry.get(
                    tool_call.name
                )

                result = tool.execute(
                    **tool_call.arguments
                )

                messages.append(
                    {
                        "role": "tool",
                        "content": str(result),
                        "tool_name": tool_call.name,
                    }
                )