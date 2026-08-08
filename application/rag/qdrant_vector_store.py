from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from application.rag.chunk import Chunk
from application.rag.similarity import cosine_similarity
from application.rag.vector_store import VectorStore
import os


class QdrantVectorStore(VectorStore):

    def __init__(
            self,
            collection_name: str,
            vector_size: int,
            host: str | None = None,
            port: int = 6333,
    ):
        self.collection_name = collection_name
        host = host or os.getenv(
            "QDRANT_HOST",
            "localhost",
        )

        self.client = QdrantClient(
            host=host,
            port=port,
        )

        self._ensure_collection(
            vector_size
        )

    def _ensure_collection(
            self,
            vector_size: int,
    ) -> None:
        collections = (
            self.client.get_collections()
        )

        exists = any(
            collection.name
            == self.collection_name
            for collection in collections.collections
        )

        if exists:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    def add(
            self,
            chunk: Chunk,
            embedding: list[float],
    ) -> None:
        point = PointStruct(
            id=hash(
                chunk.content
            ) & 0xFFFFFFFF,
            vector=embedding,
            payload={
                "content": chunk.content,
                "metadata": chunk.metadata,
            },
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )

    def search(
            self,
            query_embedding: list[float],
            top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )

        return [
            (
                Chunk(
                    content=result.payload["content"],
                    metadata=result.payload.get(
                        "metadata",
                        {},
                    ),
                ),
                result.score,
            )
            for result in response.points
        ]
