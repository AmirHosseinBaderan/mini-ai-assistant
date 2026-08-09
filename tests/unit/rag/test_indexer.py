from application.rag.chunk import Chunk
from application.rag.embedding import EmbeddingProvider
from application.rag.in_memory_vector_store import InMemoryVectorStore
from application.rag.indexer import Indexer


class FakeEmbeddingProvider(EmbeddingProvider):

    def embed(self, text: str) -> list[float]:
        return [
            float(len(text)),
            1.0,
        ]

def test_index_chunks():

    embedding_provider = FakeEmbeddingProvider()
    vector_store = InMemoryVectorStore()

    indexer = Indexer(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    chunks = [
        Chunk(
            content="Python",
        ),
        Chunk(
            content="Machine Learning",
        ),
    ]

    indexer.index(chunks)

    results = vector_store.search(
        [6.0, 1.0],
        top_k=2,
    )

    assert len(results) == 2

    assert results[0][0].content == "Python"

def test_index_empty_chunks():

    embedding_provider = FakeEmbeddingProvider()
    vector_store = InMemoryVectorStore()

    indexer = Indexer(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    indexer.index([])

    results = vector_store.search(
        [1.0, 0.0],
    )

    assert results == []