import torch

from intent_classifier.encoder import (
    EncoderBlock,TransformerEncoder
)


def test_encoder_block_shape():

    block = EncoderBlock(
        embed_dim=128,
        num_heads=4,
        ff_dim=256,
        dropout=0.1,
    )

    x = torch.randn(
        4,
        64,
        128,
    )

    mask = torch.ones(
        4,
        64,
        dtype=torch.long,
    )

    output = block(
        x,
        mask,
    )

    assert output.shape == (
        4,
        64,
        128,
    )


def test_encoder_block_with_padding():

    block = EncoderBlock(
        embed_dim=128,
        num_heads=4,
        ff_dim=256,
    )

    x = torch.randn(
        2,
        8,
        128,
    )

    mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )

    output = block(
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


def test_encoder_block_backward():

    block = EncoderBlock(
        embed_dim=128,
        num_heads=4,
        ff_dim=256,
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

    output = block(
        x,
        mask,
    )

    loss = output.mean()

    loss.backward()

    assert x.grad is not None

    assert torch.isfinite(
        x.grad
    ).all()
    
def test_transformer_encoder_shape():

    encoder = TransformerEncoder(
        num_layers=2,
        embed_dim=128,
        num_heads=4,
        ff_dim=256,
        dropout=0.1,
    )

    x = torch.randn(
        4,
        64,
        128,
    )

    mask = torch.ones(
        4,
        64,
        dtype=torch.long,
    )

    output = encoder(
        x,
        mask,
    )

    assert output.shape == (
        4,
        64,
        128,
    )
    
def test_transformer_encoder_backward():

    encoder = TransformerEncoder(
        num_layers=2,
        embed_dim=128,
        num_heads=4,
        ff_dim=256,
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

    output = encoder(
        x,
        mask,
    )

    loss = output.mean()

    loss.backward()

    assert x.grad is not None

    assert torch.isfinite(
        x.grad
    ).all()