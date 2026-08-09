import pytest

from application.rag.similarity import cosine_similarity


def test_identical_vectors():
    result = cosine_similarity(
        [1.0, 0.0],
        [1.0, 0.0],
    )

    assert result == pytest.approx(1.0)


def test_orthogonal_vectors():
    result = cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    )

    assert result == pytest.approx(0.0)


def test_opposite_vectors():
    result = cosine_similarity(
        [1.0, 0.0],
        [-1.0, 0.0],
    )

    assert result == pytest.approx(-1.0)
    
def test_cosine_similarity():
    result = cosine_similarity(
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    )

    assert result == pytest.approx(
        0.974631846,
    )
    
def test_different_dimensions():
    with pytest.raises(ValueError):
        cosine_similarity(
            [1.0, 2.0],
            [1.0],
        )


def test_empty_vectors():
    with pytest.raises(ValueError):
        cosine_similarity(
            [],
            [],
        )


def test_zero_vector():
    with pytest.raises(ValueError):
        cosine_similarity(
            [0.0, 0.0],
            [1.0, 2.0],
        )