import pytest

from application.rag.chunk import Chunk
from application.rag.in_memory_vector_store import InMemoryVectorStore


def test_add_and_search():

    store = InMemoryVectorStore()

    first = Chunk(
        content="Python programming",
    )

    second = Chunk(
        content="Machine learning",
    )

    store.add(
        first,
        [1.0, 0.0],
    )

    store.add(
        second,
        [0.0, 1.0],
    )

    results = store.search(
        [1.0, 0.0],
    )

    assert results[0][0] == first
    assert results[0][1] == 1.0
    
def test_search_returns_results_by_similarity():

    store = InMemoryVectorStore()

    first = Chunk(
        content="Very similar",
    )

    second = Chunk(
        content="Somewhat similar",
    )

    third = Chunk(
        content="Different",
    )

    store.add(
        first,
        [1.0, 0.0],
    )

    store.add(
        second,
        [0.8, 0.2],
    )

    store.add(
        third,
        [0.0, 1.0],
    )

    results = store.search(
        [1.0, 0.0],
        top_k=3,
    )

    assert results[0][0] == first
    assert results[1][0] == second
    assert results[2][0] == third
    
def test_search_returns_results_by_similarity():

    store = InMemoryVectorStore()

    first = Chunk(
        content="Very similar",
    )

    second = Chunk(
        content="Somewhat similar",
    )

    third = Chunk(
        content="Different",
    )

    store.add(
        first,
        [1.0, 0.0],
    )

    store.add(
        second,
        [0.8, 0.2],
    )

    store.add(
        third,
        [0.0, 1.0],
    )

    results = store.search(
        [1.0, 0.0],
        top_k=3,
    )

    assert results[0][0] == first
    assert results[1][0] == second
    assert results[2][0] == third
    
def test_invalid_top_k():

    store = InMemoryVectorStore()

    with pytest.raises(ValueError):
        store.search(
            [1.0, 0.0],
            top_k=0,
        )