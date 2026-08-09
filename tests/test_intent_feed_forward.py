import torch

from intent_classifier.feed_forward import (
    FeedForward,
)


def test_feed_forward_shape():

    feed_forward = FeedForward(
        embed_dim=128,
        ff_dim=256,
    )

    x = torch.randn(
        4,
        64,
        128,
    )

    output = feed_forward(x)

    assert output.shape == (
        4,
        64,
        128,
    )


def test_feed_forward_backward():

    feed_forward = FeedForward(
        embed_dim=128,
        ff_dim=256,
    )

    x = torch.randn(
        2,
        8,
        128,
        requires_grad=True,
    )

    output = feed_forward(x)

    loss = output.mean()

    loss.backward()

    assert x.grad is not None

    assert torch.isfinite(
        x.grad
    ).all()