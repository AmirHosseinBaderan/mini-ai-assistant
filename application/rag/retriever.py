from application.rag.chunk import Chunk
from application.rag.embedding import EmbeddingProvider
from application.rag.vector_store import VectorStore


class Retriever:

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def retrieve(
            self,
            query: str,
            top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        if not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        query_embedding = self.embedding_provider.embed(
            query
        )

        return self.vector_store.search(
            query_embedding,
            top_k=top_k,
        )