from application.rag.chunk import Chunk
from application.rag.embedding import EmbeddingProvider
from application.rag.vector_store import VectorStore


class Indexer:

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def index(
        self,
        chunks: list[Chunk],
    ) -> None:

        for chunk in chunks:
            embedding = self.embedding_provider.embed(
                chunk.content
            )

            self.vector_store.add(
                chunk,
                embedding,
            )