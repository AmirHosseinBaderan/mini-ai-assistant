from abc import ABC, abstractmethod
from typing import Any

class Tool(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def execute(
        self,
        **kwargs: Any,
    ) -> Any:
        ...