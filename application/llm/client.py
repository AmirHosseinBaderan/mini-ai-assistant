from abc import ABC, abstractmethod
from typing import Iterator

from application.llm.response import LLMResponse


class LLMClient(ABC):

    @abstractmethod
    def stream(
        self,
        messages: list[dict[str, str]],
    ) -> Iterator[str]:
        pass

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
    ) -> LLMResponse:
        pass

    @abstractmethod
    def embed(
        self,
        text: str,
    ) -> list[float]:
        pass