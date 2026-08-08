import os
from typing import Iterator

from ollama import Client

from application.llm.client import LLMClient


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
