import tempfile
from pathlib import Path

import torch

from intent_classifier.checkpoint import CheckpointManager
from intent_classifier.early_stopping import EarlyStopping


def test_checkpoint_manager_defaults():
    with tempfile.TemporaryDirectory() as tmpdir:
        cm = CheckpointManager(
            checkpoint_dir=tmpdir,
            monitor="validation_f1",
            mode="max",
        )

        assert cm.monitor == "validation_f1"
        assert cm.mode == "max"
        assert cm.best_value == float("-inf")
        assert Path(tmpdir).exists()


def test_checkpoint_manager_is_better_max():
    with tempfile.TemporaryDirectory() as tmpdir:
        cm = CheckpointManager(
            checkpoint_dir=tmpdir,
            monitor="validation_f1",
            mode="max",
        )

        assert cm.is_better(0.5) is True
        cm.update_best(0.5)
        assert cm.is_better(0.6) is True
        assert cm.is_better(0.4) is False
        assert cm.is_better(0.5) is False


def test_checkpoint_manager_is_better_min():
    with tempfile.TemporaryDirectory() as tmpdir:
        cm = CheckpointManager(
            checkpoint_dir=tmpdir,
            monitor="validation_loss",
            mode="min",
        )

        assert cm.is_better(0.5) is True
        cm.update_best(0.5)
        assert cm.is_better(0.4) is True
        assert cm.is_better(0.6) is False
        assert cm.is_better(0.5) is False


def test_checkpoint_manager_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        cm = CheckpointManager(
            checkpoint_dir=tmpdir,
            monitor="validation_f1",
            mode="max",
        )

        model_state = {"weight": torch.tensor([1.0, 2.0])}
        optimizer_state = {"lr": 3e-4}
        metrics = {
            "epoch": 1,
            "validation_f1": 0.85,
            "train_loss": 0.5,
        }

        best_path = cm.save_best(
            epoch=1,
            model_state_dict=model_state,
            optimizer_state_dict=optimizer_state,
            metrics=metrics,
        )

        assert best_path.exists()

        last_path = cm.save_last(
            epoch=2,
            model_state_dict=model_state,
            optimizer_state_dict=optimizer_state,
            metrics=metrics,
        )

        assert last_path.exists()

        loaded = cm.load(best_path)

        assert loaded["epoch"] == 1
        assert loaded["metrics"]["validation_f1"] == 0.85
        assert torch.equal(
            loaded["model_state_dict"]["weight"],
            torch.tensor([1.0, 2.0]),
        )


def test_early_stopping_defaults():
    es = EarlyStopping(
        monitor="validation_f1",
        mode="max",
        patience=5,
        min_delta=0.0,
    )

    assert es.monitor == "validation_f1"
    assert es.mode == "max"
    assert es.patience == 5
    assert es.min_delta == 0.0
    assert es.wait == 0
    assert es.stop_training is False
    assert es.stopped_epoch == 0


def test_early_stopping_max_mode():
    es = EarlyStopping(
        monitor="validation_f1",
        mode="max",
        patience=3,
        min_delta=0.0,
    )

    assert es.step(0.5, 1) is False
    assert es.wait == 0

    assert es.step(0.4, 2) is False
    assert es.wait == 1

    assert es.step(0.3, 3) is False
    assert es.wait == 2

    assert es.step(0.2, 4) is True
    assert es.stop_training is True
    assert es.stopped_epoch == 4


def test_early_stopping_min_mode():
    es = EarlyStopping(
        monitor="validation_loss",
        mode="min",
        patience=3,
        min_delta=0.0,
    )

    assert es.step(0.5, 1) is False
    assert es.wait == 0

    assert es.step(0.6, 2) is False
    assert es.wait == 1

    assert es.step(0.7, 3) is False
    assert es.wait == 2

    assert es.step(0.8, 4) is True
    assert es.stop_training is True
    assert es.stopped_epoch == 4


def test_early_stopping_min_delta():
    es = EarlyStopping(
        monitor="validation_f1",
        mode="max",
        patience=3,
        min_delta=0.1,
    )

    assert es.step(0.5, 1) is False
    assert es.wait == 0

    assert es.step(0.54, 2) is False
    assert es.wait == 1

    assert es.step(0.55, 3) is False
    assert es.wait == 2

    assert es.step(0.56, 4) is True
    assert es.stop_training is True


def test_early_stopping_reset_on_improvement():
    es = EarlyStopping(
        monitor="validation_f1",
        mode="max",
        patience=3,
        min_delta=0.0,
    )

    assert es.step(0.5, 1) is False
    assert es.wait == 0

    assert es.step(0.4, 2) is False
    assert es.wait == 1

    assert es.step(0.6, 3) is False
    assert es.wait == 0

    assert es.step(0.5, 4) is False
    assert es.wait == 1

    assert es.step(0.4, 5) is False
    assert es.wait == 2

    assert es.step(0.3, 6) is True
    assert es.stop_training is True
