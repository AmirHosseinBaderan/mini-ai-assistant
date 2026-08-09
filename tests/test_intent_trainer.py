import torch
from torch.utils.data import DataLoader, TensorDataset

from intent_classifier.trainer import (
    IntentTrainer,
)


class TinyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = torch.nn.Embedding(
            20,
            8,
        )

        self.classifier = torch.nn.Linear(
            8,
            2,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
    ):

        x = self.embedding(
            input_ids
        )

        mask = attention_mask.unsqueeze(-1)

        x = (
            x * mask
        ).sum(dim=1)

        count = mask.sum(
            dim=1
        ).clamp_min(1)

        x = x / count

        return self.classifier(x)


def create_loader():

    input_ids = torch.tensor([
        [1, 2, 0, 0],
        [3, 4, 5, 0],
        [6, 7, 8, 9],
        [1, 3, 0, 0],
        [4, 5, 6, 0],
        [7, 8, 9, 1],
    ])

    attention_mask = torch.tensor([
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ])

    labels = torch.tensor([
        0,
        1,
        1,
        0,
        1,
        0,
    ])

    dataset = TensorDataset(
        input_ids,
        attention_mask,
        labels,
    )

    def collate_fn(batch):

        input_ids = torch.stack([
            item[0]
            for item in batch
        ])

        attention_mask = torch.stack([
            item[1]
            for item in batch
        ])

        labels = torch.stack([
            item[2]
            for item in batch
        ])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": labels,
        }

    return DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
    )


def test_trainer_train_epoch():

    train_loader = create_loader()

    validation_loader = create_loader()

    model = TinyModel()

    trainer = IntentTrainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        learning_rate=3e-4,
    )

    metrics = trainer.train_epoch()

    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics

    assert torch.isfinite(
        torch.tensor(metrics["loss"])
    )


def test_trainer_validate():

    train_loader = create_loader()

    validation_loader = create_loader()

    model = TinyModel()

    trainer = IntentTrainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
    )

    metrics = trainer.validate()

    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics

    assert torch.isfinite(
        torch.tensor(metrics["loss"])
    )


def test_trainer_fit():

    train_loader = create_loader()

    validation_loader = create_loader()

    model = TinyModel()

    trainer = IntentTrainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
    )

    history, best_epoch = trainer.fit(
        epochs=2,
    )

    assert len(history) == 2

    assert history[0]["epoch"] == 1
    assert history[1]["epoch"] == 2

    assert "train_loss" in history[0]
    assert "validation_loss" in history[0]
    assert "validation_f1" in history[0]