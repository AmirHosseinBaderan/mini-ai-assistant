import pytest

from application.rag.embedding import EmbeddingProvider


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
    
def test_embedding_provider_is_abstract():
    with pytest.raises(TypeError):
        EmbeddingProvider()