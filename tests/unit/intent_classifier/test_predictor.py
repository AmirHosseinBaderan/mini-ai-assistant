import tempfile
from pathlib import Path

import torch

from intent_classifier.checkpoint import CheckpointManager
from intent_classifier.config import IntentConfig
from intent_classifier.labels import IntentLabel
from intent_classifier.model import IntentClassifier
from intent_classifier.predictor import IntentPredictor, ClassificationResult
from intent_classifier.tokenizer import IntentTokenizer


def create_tokenizer():
    tokenizer = IntentTokenizer()
    tokenizer.build_vocab([
        "hello",
        "world",
        "python",
        "is",
        "great",
        "how",
        "are",
        "you",
        "what",
        "up",
    ])
    return tokenizer


def create_checkpoint(tmpdir, tokenizer):
    model = IntentClassifier(
        vocab_size=tokenizer.vocab_size,
        num_classes=2,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        ff_dim=32,
        max_length=8,
        dropout=0.0,
    )

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=tmpdir,
        monitor="validation_f1",
        mode="max",
    )

    checkpoint_manager.save_last(
        epoch=1,
        model_state_dict=model.state_dict(),
        optimizer_state_dict={},
        metrics={
            "epoch": 1,
            "best_epoch": 1,
            "validation_f1": 0.9,
            "train_loss": 0.1,
        },
    )

    checkpoint_manager.save_best(
        epoch=1,
        model_state_dict=model.state_dict(),
        optimizer_state_dict={},
        metrics={
            "epoch": 1,
            "best_epoch": 1,
            "validation_f1": 0.9,
            "train_loss": 0.1,
        },
    )

    return model


def test_predictor_loads_best_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer = create_tokenizer()
        create_checkpoint(tmpdir, tokenizer)

        config = IntentConfig(
            embed_dim=16,
            num_heads=2,
            num_layers=1,
            ff_dim=32,
            max_length=8,
            num_classes=2,
        )

        predictor = IntentPredictor(
            tokenizer=tokenizer,
            config=config,
            checkpoint_dir=tmpdir,
            use_best=True,
        )

        assert predictor.model is not None
        assert predictor.device is not None


def test_predictor_loads_last_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer = create_tokenizer()
        create_checkpoint(tmpdir, tokenizer)

        config = IntentConfig(
            embed_dim=16,
            num_heads=2,
            num_layers=1,
            ff_dim=32,
            max_length=8,
            num_classes=2,
        )

        predictor = IntentPredictor(
            tokenizer=tokenizer,
            config=config,
            checkpoint_dir=tmpdir,
            use_best=False,
        )

        assert predictor.model is not None


def test_predictor_predict_returns_result():
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer = create_tokenizer()
        create_checkpoint(tmpdir, tokenizer)

        config = IntentConfig(
            embed_dim=16,
            num_heads=2,
            num_layers=1,
            ff_dim=32,
            max_length=8,
            num_classes=2,
        )

        predictor = IntentPredictor(
            tokenizer=tokenizer,
            config=config,
            checkpoint_dir=tmpdir,
        )

        result = predictor.predict("hello world")

        assert isinstance(result, ClassificationResult)
        assert isinstance(result.label, str)
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0
        assert result.label in ("chat", "rag")


def test_predictor_uses_config_checkpoint_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer = create_tokenizer()
        create_checkpoint(tmpdir, tokenizer)

        config = IntentConfig(
            embed_dim=16,
            num_heads=2,
            num_layers=1,
            ff_dim=32,
            max_length=8,
            num_classes=2,
            checkpoint_dir=Path(tmpdir),
        )

        predictor = IntentPredictor(
            tokenizer=tokenizer,
            config=config,
        )

        assert predictor.model is not None


def test_predictor_defaults_to_best_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer = create_tokenizer()
        create_checkpoint(tmpdir, tokenizer)

        config = IntentConfig(
            embed_dim=16,
            num_heads=2,
            num_layers=1,
            ff_dim=32,
            max_length=8,
            num_classes=2,
            checkpoint_dir=Path(tmpdir),
        )

        # Default use_best=True
        predictor = IntentPredictor(
            tokenizer=tokenizer,
            config=config,
        )

        assert predictor.model is not None
