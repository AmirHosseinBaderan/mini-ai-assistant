from abc import ABC, abstractmethod
from collections.abc import Iterator

from application.llm.message import LLMMessage
from application.llm.response import LLMResponse
from application.llm.stream_event import LLMStreamEvent
from application.llm.tool import LLMTool


class LLMClient(ABC):

    @abstractmethod
    def chat(
        self,
        messages: list[LLMMessage],
        tools: list[LLMTool],
    ) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    def stream(
        self,
        messages: list[LLMMessage],
    ) -> Iterator[str]:
        raise NotImplementedError

    @abstractmethod
    def stream_chat(
        self,
        messages: list[LLMMessage],
        tools: list[LLMTool],
    ) -> Iterator[LLMStreamEvent]:
        raise NotImplementedError