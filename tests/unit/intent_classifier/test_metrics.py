import torch

from intent_classifier.metrics import (
    accuracy,
    f1_score,
    precision,
    recall,
)


def test_accuracy():

    predictions = torch.tensor(
        [0, 1, 1, 0]
    )

    labels = torch.tensor(
        [0, 1, 0, 0]
    )

    assert accuracy(
        predictions,
        labels,
    ) == 0.75


def test_precision():

    predictions = torch.tensor(
        [0, 1, 1, 0]
    )

    labels = torch.tensor(
        [0, 1, 0, 0]
    )

    assert precision(
        predictions,
        labels,
    ) == 0.5


def test_recall():

    predictions = torch.tensor(
        [0, 1, 1, 0]
    )

    labels = torch.tensor(
        [0, 1, 0, 0]
    )

    assert recall(
        predictions,
        labels,
    ) == 1.0


def test_f1():

    predictions = torch.tensor(
        [0, 1, 1, 0]
    )

    labels = torch.tensor(
        [0, 1, 0, 0]
    )

    assert abs(
        f1_score(
            predictions,
            labels,
        ) - (2 / 3)
    ) < 1e-6