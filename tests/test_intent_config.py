import torch

from intent_classifier.config import (
    IntentConfig,
)


def test_default_config():

    config = IntentConfig()

    assert config.max_length == 64

    assert config.embed_dim == 128

    assert config.num_heads == 4

    assert config.num_layers == 2

    assert config.ff_dim == 256

    assert config.dropout == 0.1

    assert config.num_classes == 2

    assert config.batch_size == 32

    assert config.learning_rate == 3e-4

    assert config.weight_decay == 1e-2

    assert config.epochs == 20


def test_device():

    config = IntentConfig()

    assert isinstance(
        config.device,
        torch.device,
    )

    assert config.device.type in (
        "cpu",
        "cuda",
    )