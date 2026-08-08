from application.rag.chunk import Chunk
from application.rag.embedding import EmbeddingProvider
from application.rag.in_memory_vector_store import InMemoryVectorStore
from application.rag.retriever import Retriever
import pytest

class FakeEmbeddingProvider(EmbeddingProvider):

    def embed(self, text: str) -> list[float]:
        embeddings = {
            "python": [1.0, 0.0],
            "machine learning": [0.0, 1.0],
            "programming": [0.9, 0.1],
        }

        return embeddings[text]

def test_retriever_returns_most_relevant_chunks():

    embedding_provider = FakeEmbeddingProvider()
    vector_store = InMemoryVectorStore()

    python_chunk = Chunk(
        content="Python is a programming language."
    )

    ml_chunk = Chunk(
        content="Machine learning uses data."
    )

    programming_chunk = Chunk(
        content="Programming involves writing code."
    )

    vector_store.add(
        python_chunk,
        embedding_provider.embed("python"),
    )

    vector_store.add(
        ml_chunk,
        embedding_provider.embed("machine learning"),
    )

    vector_store.add(
        programming_chunk,
        embedding_provider.embed("programming"),
    )

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        "python",
        top_k=2,
    )

    assert len(results) == 2

    assert results[0][0] == python_chunk
    assert results[0][1] == 1.0

def test_retriever_uses_embedding_provider():

    embedding_provider = FakeEmbeddingProvider()
    vector_store = InMemoryVectorStore()

    chunk = Chunk(
        content="Python"
    )

    vector_store.add(
        chunk,
        [1.0, 0.0],
    )

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        "python",
        top_k=1,
    )

    assert results[0][0] == chunk

def test_empty_query_is_rejected():

    embedding_provider = FakeEmbeddingProvider()
    vector_store = InMemoryVectorStore()

    retriever = Retriever(
        embedding_provider,
        vector_store,
    )

    with pytest.raises(ValueError):
        retriever.retrieve("")