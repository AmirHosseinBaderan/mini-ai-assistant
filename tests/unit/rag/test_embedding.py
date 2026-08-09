from application.rag.embedding import EmbeddingProvider
from application.rag.ollama_embedding import OllamaEmbeddingProvider
import pytest
from application.llm.ollama_client import  OllamaClient

class FakeEmbeddingProvider(EmbeddingProvider):

    def embed(self, text: str) -> list[float]:
        return [
            float(len(text)),
            1.0,
            0.5,
        ]


def test_embed():
    provider = FakeEmbeddingProvider()

    vector = provider.embed("hello")

    assert vector == [
        5.0,
        1.0,
        0.5,
    ]


def test_embed_many():
    provider = FakeEmbeddingProvider()

    vectors = provider.embed_many(
        [
            "hello",
            "world",
        ]
    )

    assert vectors == [
        [5.0, 1.0, 0.5],
        [5.0, 1.0, 0.5],
    ]

@pytest.mark.integration
def test_ollama_embedding():

    client = OllamaClient()

    provider = OllamaEmbeddingProvider(
        client=client
    )

    vector = provider.embed(
        "Python is a programming language."
    )

    assert isinstance(vector, list)
    assert len(vector) > 0
    assert all(
        isinstance(value, float)
        for value in vector
    )