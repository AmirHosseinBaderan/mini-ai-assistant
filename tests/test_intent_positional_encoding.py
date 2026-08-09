import torch

from intent_classifier.positional_encoding import (
    PositionalEncoding,
)


def test_positional_encoding_shape():

    positional_encoding = PositionalEncoding(
        embed_dim=128,
        max_length=64,
    )

    x = torch.zeros(
        4,
        64,
        128,
    )

    output = positional_encoding(x)

    assert output.shape == (
        4,
        64,
        128,
    )


def test_positional_encoding_changes_input():

    positional_encoding = PositionalEncoding(
        embed_dim=128,
        max_length=64,
    )

    x = torch.zeros(
        2,
        8,
        128,
    )

    output = positional_encoding(x)

    assert not torch.equal(
        x,
        output,
    )


def test_positional_encoding_does_not_modify_input():

    positional_encoding = PositionalEncoding(
        embed_dim=128,
        max_length=64,
    )

    x = torch.randn(
        2,
        8,
        128,
    )

    original = x.clone()

    positional_encoding(x)

    assert torch.equal(
        x,
        original,
    )


def test_sequence_length_limit():

    positional_encoding = PositionalEncoding(
        embed_dim=128,
        max_length=64,
    )

    x = torch.randn(
        2,
        65,
        128,
    )

    try:

        positional_encoding(x)

        assert False

    except ValueError:
        assert True