import os
from typing import Any

from ollama import Client

from application.llm.client import LLMClient
from application.llm.message import LLMMessage
from application.llm.response import LLMResponse, ToolCall
from application.llm.tool import LLMTool

from collections.abc import Iterator

from application.llm.stream_event import LLMStreamEvent


class OllamaClient(LLMClient):

    def __init__(
            self,
            host: str | None = None,
            model: str | None = None,
            embedding_model: str | None = None,
    ):
        self.host = host or os.getenv(
            "OLLAMA_HOST",
            "http://localhost:11434",
        )

        self.model = model or os.getenv(
            "OLLAMA_MODEL",
            "llama3.2:1b",
        )

        self.embedding_model = embedding_model or os.getenv(
            "OLLAMA_EMBEDDING_MODEL",
            "nomic-embed-text:latest",
        )

        self.client = Client(
            host=self.host,
        )

    def stream(
            self,
            messages: list[LLMMessage],
    ) -> Iterator[str]:

        response = self.client.chat(
            model=self.model,
            messages=[
                self._to_ollama_message(message)
                for message in messages
            ],
            stream=True,
        )

        for chunk in response:

            content = chunk["message"]["content"]

            if content:
                yield content

    def chat(
            self,
            messages: list[LLMMessage],
            tools: list[LLMTool],
    ) -> LLMResponse:

        response = self.client.chat(
            model=self.model,
            messages=[
                self._to_ollama_message(message)
                for message in messages
            ],
            tools=[
                self._to_ollama_tool(tool)
                for tool in tools
            ],
            stream=False,
        )

        message = response.message

        tool_calls = []

        for tool_call in message.tool_calls or []:
            tool_calls.append(
                ToolCall(
                    name=tool_call.function.name,
                    arguments=dict(
                        tool_call.function.arguments
                    ),
                )
            )

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls or None,
        )

    def embed(
            self,
            text: str,
    ) -> list[float]:

        response = self.client.embed(
            model=self.embedding_model,
            input=text,
        )

        return response["embeddings"][0]

    def _to_ollama_message(
            self,
            message: LLMMessage,
    ) -> dict[str, Any]:

        result: dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }

        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    }
                }
                for tool_call in message.tool_calls
            ]

        if message.tool_name:
            result["tool_name"] = message.tool_name

        return result

    def _to_ollama_tool(
            self,
            tool: LLMTool,
    ) -> dict[str, Any]:

        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    def stream_chat(
            self,
            messages: list[LLMMessage],
            tools: list[LLMTool],
    ) -> Iterator[LLMStreamEvent]:

        response = self.client.chat(
            model=self.model,
            messages=[
                self._to_ollama_message(message)
                for message in messages
            ],
            tools=[
                self._to_ollama_tool(tool)
                for tool in tools
            ],
            stream=True,
        )

        for chunk in response:

            message = chunk["message"]

            content = message.get(
                "content",
                "",
            )

            if content:
                yield LLMStreamEvent(
                    type="text",
                    content=content,
                )

            tool_calls = message.get(
                "tool_calls"
            )

            if tool_calls:

                for tool_call in tool_calls:
                    function = tool_call["function"]

                    yield LLMStreamEvent(
                        type="tool_call",
                        tool_name=function["name"],
                        tool_arguments=function["arguments"],
                    )

        yield LLMStreamEvent(
            type="done",
        )
