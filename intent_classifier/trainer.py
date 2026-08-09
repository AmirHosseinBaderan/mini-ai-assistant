from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm

from .checkpoint import CheckpointManager
from .early_stopping import EarlyStopping
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

    def train_epoch(
        self,
        epoch: int = 1,
        total_epochs: int = 1,
    ) -> dict[str, float]:

        self.model.train()

        total_loss = 0.0
        all_predictions = []
        all_labels = []

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{total_epochs} [Train]",
            leave=False,
        )

        for batch in pbar:

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
    def validate(
        self,
        epoch: int = 1,
        total_epochs: int = 1,
    ) -> dict[str, float]:

        self.model.eval()

        total_loss = 0.0
        all_predictions = []
        all_labels = []

        pbar = tqdm(
            self.validation_loader,
            desc=f"Epoch {epoch}/{total_epochs} [Val]",
            leave=False,
        )

        for batch in pbar:

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
        checkpoint_dir: str | Path | None = None,
        early_stopping: EarlyStopping | None = None,
    ) -> tuple[list[dict[str, float]], int]:
        history = []
        start_epoch = 1
        best_epoch = 0

        checkpoint_manager = None

        if checkpoint_dir is not None:
            checkpoint_manager = CheckpointManager(
                checkpoint_dir=Path(checkpoint_dir),
            )

            if checkpoint_manager.has_last_checkpoint():
                try:
                    checkpoint = checkpoint_manager.load_last()
                    self.model.load_state_dict(
                        checkpoint["model_state_dict"]
                    )
                    self.optimizer.load_state_dict(
                        checkpoint["optimizer_state_dict"]
                    )
                    start_epoch = checkpoint["metrics"]["epoch"] + 1
                    best_epoch = checkpoint["metrics"].get(
                        "best_epoch", start_epoch - 1
                    )

                    # Restore best_value from checkpoint metrics if available
                    monitor_key = checkpoint_manager.monitor
                    if monitor_key in checkpoint["metrics"]:
                        checkpoint_manager.update_best(
                            checkpoint["metrics"][monitor_key]
                        )

                    print(
                        f"Resumed from checkpoint at epoch "
                        f"{checkpoint['metrics']['epoch']}"
                    )
                except Exception as e:
                    print(
                        f"Failed to load checkpoint: {e}. "
                        f"Starting from scratch."
                    )

        epoch_pbar = tqdm(
            range(start_epoch, epochs + 1),
            desc="Epochs",
            dynamic_ncols=True,
        )

        for epoch in epoch_pbar:

            train_metrics = self.train_epoch(
                epoch=epoch,
                total_epochs=epochs,
            )

            validation_metrics = self.validate(
                epoch=epoch,
                total_epochs=epochs,
            )

            result = {
                "epoch": epoch,
                "best_epoch": best_epoch,
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

            epoch_pbar.set_postfix(
                {
                    "Train Loss": f"{result['train_loss']:.4f}",
                    "Val F1": f"{result['validation_f1']:.4f}",
                }
            )

            if checkpoint_manager is not None:
                checkpoint_manager.save_last(
                    epoch=epoch,
                    model_state_dict=self.model.state_dict(),
                    optimizer_state_dict=self.optimizer.state_dict(),
                    metrics=result,
                )

                if checkpoint_manager.is_better(
                    result[checkpoint_manager.monitor]
                ):
                    checkpoint_manager.update_best(
                        result[checkpoint_manager.monitor]
                    )

                    checkpoint_manager.save_best(
                        epoch=epoch,
                        model_state_dict=self.model.state_dict(),
                        optimizer_state_dict=self.optimizer.state_dict(),
                        metrics=result,
                    )

                    best_epoch = epoch

            if (
                early_stopping is not None
                and early_stopping.step(
                    result[early_stopping.monitor],
                    epoch,
                )
            ):
                print(
                    f"\nEarly stopping triggered at epoch {epoch}. "
                    f"Best epoch: {early_stopping.best_value:.4f}"
                )
                break

        return history, best_epoch
