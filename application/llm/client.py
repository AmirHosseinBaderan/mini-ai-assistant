from abc import ABC, abstractmethod
from typing import Iterator


class LLMClient(ABC):

    @abstractmethod
    def stream(
        self,
        messages: list[dict[str, str]],
    ) -> Iterator[str]:
        pass

    @abstractmethod
    def embed(self,text)-> list[float]:
        pass