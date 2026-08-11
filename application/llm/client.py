from abc import ABC, abstractmethod
from typing import Iterator

from application.llm.message import LLMMessage
from application.llm.response import LLMResponse
from application.llm.tool import LLMTool

class LLMClient(ABC):

    @abstractmethod
    def stream(
        self,
        messages: list[LLMMessage],
    ) -> Iterator[str]:
        pass

    @abstractmethod
    def chat(
        self,
        messages: list[LLMMessage],
        tools: list[LLMTool],
    ) -> LLMResponse:
        pass

    @abstractmethod
    def embed(
        self,
        text: str,
    ) -> list[float]:
        pass