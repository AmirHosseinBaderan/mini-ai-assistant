import pytest

from application.rag.chunk import Chunk
from application.rag.in_memory_vector_store import InMemoryVectorStore
from application.rag.ollama_embedding import OllamaEmbeddingProvider
from application.rag.retriever import Retriever
from application.llm.ollama_client import OllamaClient


@pytest.mark.integration
def test_real_semantic_retrieval():

    client = OllamaClient()

    embedding_provider = OllamaEmbeddingProvider(
        client=client
    )

    vector_store = InMemoryVectorStore()

    chunks = [
        Chunk(
            content="Python is a programming language used to build software.",
            metadata={"source": "programming.txt"},
        ),
        Chunk(
            content="Germany is a country located in Europe.",
            metadata={"source": "geography.txt"},
        ),
        Chunk(
            content="Cats are common domestic animals and popular pets.",
            metadata={"source": "animals.txt"},
        ),
    ]

    for chunk in chunks:
        embedding = embedding_provider.embed(
            chunk.content
        )

        vector_store.add(
            chunk,
            embedding
        )

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        "How can I write programs using Python?",
        top_k=3,
    )

    assert len(results) == 3

    assert results[0][0].content == (
        "Python is a programming language used to build software."
    )

    assert results[0][1] > results[1][1]