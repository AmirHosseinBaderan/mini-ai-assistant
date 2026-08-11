import os
from typing import Iterator

from ollama import Client

from application.llm.client import LLMClient
from application.llm.response import LLMResponse, ToolCall


class OllamaClient(LLMClient):
    def __init__(self, host: str | None = None, model: str | None = None,embedding_model: str | None = None):
        self.host = host or os.getenv(
            "OLLAMA_HOST",
            "http://localhost:11434"
        )

        self.model = model or os.getenv(
            "OLLAMA_MODEL",
            "llama3.2:1b"
        )
        self.embedding_model = embedding_model or os.getenv(
            "OLLAMA_EMBEDDING_MODEL",
            "nomic-embed-text:latest"
        )

        self.client = Client(host=self.host)

    def stream(
            self,
            messages: list[dict[str, str]]
    ) -> Iterator[str]:
        response = self.client.chat(
            model=self.model,
            messages=messages,
            stream=True
        )

        for chunk in response:
            content = chunk["message"]["content"]
            if content:
                yield content

    def embed(self, text) -> list[float]:
        response = self.client.embed(
            model=self.embedding_model,
            input=text,
        )

        return response["embeddings"][0]

    def chat(
            self,
            messages: list[dict[str, str]],
            tools: list[dict],
    ) -> LLMResponse:

        response = self.client.chat(
            model=self.model,
            messages=messages,
            tools=tools,
            stream=False,
        )

        tool_calls = []

        for tool_call in response.message.tool_calls or []:
            tool_calls.append(
                ToolCall(
                    name=tool_call.function.name,
                    arguments=dict(
                        tool_call.function.arguments
                    ),
                )
            )

        return LLMResponse(
            content=response.message.content,
            tool_calls=tool_calls or None,
        )
