import pytest

from application.llm.ollama_client import OllamaClient
from application.rag.chunk import Chunk
from application.rag.ollama_embedding import (
    OllamaEmbeddingProvider,
)
from application.rag.qdrant_vector_store import (
    QdrantVectorStore,
)


@pytest.mark.integration
def test_qdrant_vector_store():

    client = OllamaClient()

    embedding_provider = OllamaEmbeddingProvider(
        client=client,
    )

    test_embedding = embedding_provider.embed(
        "test",
    )

    store = QdrantVectorStore(
        collection_name="mini_ai_test",
        vector_size=len(test_embedding),
    )

    chunk = Chunk(
        content="Python is a programming language.",
        metadata={
            "source": "test.txt",
        },
    )

    embedding = embedding_provider.embed(
        chunk.content,
    )

    store.add(
        chunk,
        embedding,
    )

    results = store.search(
        embedding,
        top_k=1,
    )

    assert len(results) == 1

    result_chunk, score = results[0]

    assert (
        result_chunk.content
        == chunk.content
    )

    assert (
        result_chunk.metadata["source"]
        == "test.txt"
    )

    assert score > 0.9