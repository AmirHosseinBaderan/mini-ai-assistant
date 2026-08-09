import torch

from intent_classifier.attention import (
    MultiHeadSelfAttention,
)


def test_attention_shape():

    batch_size = 4
    sequence_length = 64
    embed_dim = 128
    num_heads = 4

    attention = MultiHeadSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
    )

    x = torch.randn(
        batch_size,
        sequence_length,
        embed_dim,
    )

    mask = torch.ones(
        batch_size,
        sequence_length,
        dtype=torch.long,
    )

    output = attention(
        x,
        mask,
    )

    assert output.shape == (
        batch_size,
        sequence_length,
        embed_dim,
    )
    
def test_attention_with_padding_mask():

    attention = MultiHeadSelfAttention(
        embed_dim=128,
        num_heads=4,
    )

    x = torch.randn(
        2,
        8,
        128,
    )

    mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )

    output = attention(
        x,
        mask,
    )

    assert output.shape == (
        2,
        8,
        128,
    )

    assert torch.isfinite(
        output
    ).all()
    
def test_invalid_embedding_dimension():

    attention = MultiHeadSelfAttention(
        embed_dim=128,
        num_heads=4,
    )

    x = torch.randn(
        2,
        8,
        64,
    )

    mask = torch.ones(
        2,
        8,
        dtype=torch.long,
    )

    try:
        attention(x, mask)
        assert False

    except ValueError:
        assert True
        
def test_invalid_number_of_heads():

    try:

        MultiHeadSelfAttention(
            embed_dim=128,
            num_heads=3,
        )

        assert False

    except ValueError:
        assert True
        
def test_attention_backward():

    attention = MultiHeadSelfAttention(
        embed_dim=128,
        num_heads=4,
    )

    x = torch.randn(
        2,
        8,
        128,
        requires_grad=True,
    )

    mask = torch.ones(
        2,
        8,
        dtype=torch.long,
    )

    output = attention(
        x,
        mask,
    )

    loss = output.mean()

    loss.backward()

    assert x.grad is not None

    assert torch.isfinite(
        x.grad
    ).all()