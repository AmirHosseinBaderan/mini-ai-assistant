import torch


def accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> float:

    correct = (
        predictions == labels
    ).sum().item()

    total = labels.numel()

    if total == 0:
        return 0.0

    return correct / total


def precision(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> float:

    true_positive = (
        (predictions == 1)
        & (labels == 1)
    ).sum().item()

    predicted_positive = (
        predictions == 1
    ).sum().item()

    if predicted_positive == 0:
        return 0.0

    return true_positive / predicted_positive


def recall(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> float:

    true_positive = (
        (predictions == 1)
        & (labels == 1)
    ).sum().item()

    actual_positive = (
        labels == 1
    ).sum().item()

    if actual_positive == 0:
        return 0.0

    return true_positive / actual_positive


def f1_score(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> float:

    p = precision(
        predictions,
        labels,
    )

    r = recall(
        predictions,
        labels,
    )

    if p + r == 0:
        return 0.0

    return 2 * p * r / (p + r)