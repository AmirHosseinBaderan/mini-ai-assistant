from abc import ABC, abstractmethod

from application.rag.chunk import Chunk


class VectorStore(ABC):

    @abstractmethod
    def add(
        self,
        chunk: Chunk,
        embedding: list[float],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        raise NotImplementedError