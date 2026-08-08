import pytest

from application.llm.ollama_client import OllamaClient
from application.rag.chunk import Chunk
from application.rag.in_memory_vector_store import InMemoryVectorStore
from application.rag.indexer import Indexer
from application.rag.ollama_embedding import OllamaEmbeddingProvider
from application.rag.retriever import Retriever


@pytest.mark.integration
def test_real_indexing_and_retrieval():

    client = OllamaClient()

    embedding_provider = OllamaEmbeddingProvider(
        client=client,
    )

    vector_store = InMemoryVectorStore()

    indexer = Indexer(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    chunks = [
        Chunk(
            content="Python is a programming language.",
            metadata={"source": "programming.txt"},
        ),
        Chunk(
            content="Germany is located in Europe.",
            metadata={"source": "geography.txt"},
        ),
        Chunk(
            content="Cats are popular domestic animals.",
            metadata={"source": "animals.txt"},
        ),
    ]

    indexer.index(chunks)

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        "How can I program using Python?",
        top_k=1,
    )

    assert len(results) == 1

    assert (
        results[0][0].metadata["source"]
        == "programming.txt"
    )