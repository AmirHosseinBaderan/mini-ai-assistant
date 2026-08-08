from application.rag.chunk import Chunk
from application.rag.similarity import cosine_similarity
from application.rag.vector_store import VectorStore


class InMemoryVectorStore(VectorStore):

    def __init__(self):
        self._items: list[
            tuple[Chunk, list[float]]
        ] = []

    def add(
        self,
        chunk: Chunk,
        embedding: list[float],
    ) -> None:

        self._items.append(
            (chunk, embedding)
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        results = []

        for chunk, embedding in self._items:
            score = cosine_similarity(
                query_embedding,
                embedding,
            )

            results.append(
                (chunk, score)
            )

        results.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return results[:top_k]