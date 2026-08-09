from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

from .metrics import (
    accuracy,
    f1_score,
    precision,
    recall,
)


class IntentTrainer:

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        validation_loader,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-2,
        device: torch.device | None = None,
    ):
        self.model = model

        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self.device = device or torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def train_epoch(self) -> dict[str, float]:

        self.model.train()

        total_loss = 0.0
        all_predictions = []
        all_labels = []

        for batch in self.train_loader:

            input_ids = batch[
                "input_ids"
            ].to(self.device)

            attention_mask = batch[
                "attention_mask"
            ].to(self.device)

            labels = batch[
                "label"
            ].to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(
                input_ids,
                attention_mask,
            )

            loss = self.criterion(
                logits,
                labels,
            )

            loss.backward()

            self.optimizer.step()

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=-1,
            )

            all_predictions.append(
                predictions.detach().cpu()
            )

            all_labels.append(
                labels.detach().cpu()
            )

        predictions = torch.cat(
            all_predictions
        )

        labels = torch.cat(
            all_labels
        )

        total_samples = labels.size(0)

        return {
            "loss": total_loss / total_samples,
            "accuracy": accuracy(
                predictions,
                labels,
            ),
            "precision": precision(
                predictions,
                labels,
            ),
            "recall": recall(
                predictions,
                labels,
            ),
            "f1": f1_score(
                predictions,
                labels,
            ),
        }

    @torch.no_grad()
    def validate(self) -> dict[str, float]:

        self.model.eval()

        total_loss = 0.0
        all_predictions = []
        all_labels = []

        for batch in self.validation_loader:

            input_ids = batch[
                "input_ids"
            ].to(self.device)

            attention_mask = batch[
                "attention_mask"
            ].to(self.device)

            labels = batch[
                "label"
            ].to(self.device)

            logits = self.model(
                input_ids,
                attention_mask,
            )

            loss = self.criterion(
                logits,
                labels,
            )

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=-1,
            )

            all_predictions.append(
                predictions.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

        predictions = torch.cat(
            all_predictions
        )

        labels = torch.cat(
            all_labels
        )

        total_samples = labels.size(0)

        return {
            "loss": total_loss / total_samples,
            "accuracy": accuracy(
                predictions,
                labels,
            ),
            "precision": precision(
                predictions,
                labels,
            ),
            "recall": recall(
                predictions,
                labels,
            ),
            "f1": f1_score(
                predictions,
                labels,
            ),
        }

    def fit(
        self,
        epochs: int,
    ) -> list[dict[str, float]]:

        history = []

        for epoch in range(1, epochs + 1):

            train_metrics = self.train_epoch()

            validation_metrics = self.validate()

            result = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_precision": validation_metrics["precision"],
                "validation_recall": validation_metrics["recall"],
                "validation_f1": validation_metrics["f1"],
            }

            history.append(result)

            print(
                f"Epoch {epoch}/{epochs} "
                f"| "
                f"Train Loss: {result['train_loss']:.4f} "
                f"| "
                f"Train Acc: {result['train_accuracy']:.4f} "
                f"| "
                f"Val Loss: {result['validation_loss']:.4f} "
                f"| "
                f"Val Acc: {result['validation_accuracy']:.4f} "
                f"| "
                f"Val F1: {result['validation_f1']:.4f}"
            )

        return history

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )