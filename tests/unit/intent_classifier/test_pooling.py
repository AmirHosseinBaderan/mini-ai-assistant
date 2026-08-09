import torch

from intent_classifier.pooling import (
    MeanPooling,
)


def test_mean_pooling_shape():

    pooling = MeanPooling()

    hidden_states = torch.randn(
        4,
        64,
        128,
    )

    attention_mask = torch.ones(
        4,
        64,
        dtype=torch.long,
    )

    output = pooling(
        hidden_states,
        attention_mask,
    )

    assert output.shape == (
        4,
        128,
    )
    
def test_mean_pooling_ignores_padding():

    pooling = MeanPooling()

    hidden_states = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [100.0, 100.0],
                [100.0, 100.0],
            ]
        ]
    )

    attention_mask = torch.tensor(
        [
            [1, 1, 0, 0]
        ]
    )

    output = pooling(
        hidden_states,
        attention_mask,
    )

    expected = torch.tensor(
        [
            [2.0, 3.0]
        ]
    )

    assert torch.allclose(
        output,
        expected,
    )