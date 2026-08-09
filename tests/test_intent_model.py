import json
from pathlib import Path

import torch

from intent_classifier.dataset import (
    IntentDataset,
    create_dataloader,
)
from intent_classifier.model import IntentClassifier
from intent_classifier.tokenizer import IntentTokenizer


DATA_PATH = Path("data/intent/train.jsonl")


def load_training_texts(
    path: Path,
) -> list[str]:

    texts = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if not line.strip():
                continue

            record = json.loads(line)

            texts.append(record["text"])

    return texts


def create_test_model(
    vocab_size: int = 1017,
) -> IntentClassifier:

    return IntentClassifier(
        vocab_size=vocab_size,
        num_classes=2,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        max_length=64,
        dropout=0.1,
    )


def test_classifier_shape():

    model = create_test_model()

    input_ids = torch.randint(
        0,
        1017,
        (4, 64),
    )

    attention_mask = torch.ones(
        4,
        64,
        dtype=torch.long,
    )

    logits = model(
        input_ids,
        attention_mask,
    )

    assert logits.shape == (4, 2)

    assert torch.isfinite(logits).all()


def test_classifier_with_padding():

    model = create_test_model()

    input_ids = torch.tensor([
        [2, 0, 0, 0, 0, 0, 0, 0],
        [3, 4, 5, 6, 7, 0, 0, 0],
    ])

    attention_mask = torch.tensor([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
    ])

    logits = model(
        input_ids,
        attention_mask,
    )

    assert logits.shape == (2, 2)

    assert torch.isfinite(logits).all()


def test_classifier_loss():

    model = create_test_model()

    input_ids = torch.randint(
        0,
        1017,
        (4, 64),
    )

    attention_mask = torch.ones(
        4,
        64,
        dtype=torch.long,
    )

    labels = torch.tensor(
        [0, 1, 0, 1],
        dtype=torch.long,
    )

    logits = model(
        input_ids,
        attention_mask,
    )

    criterion = torch.nn.CrossEntropyLoss()

    loss = criterion(
        logits,
        labels,
    )

    assert loss.ndim == 0

    assert torch.isfinite(loss)


def test_classifier_backward():

    model = create_test_model()

    input_ids = torch.randint(
        0,
        1017,
        (4, 64),
    )

    attention_mask = torch.ones(
        4,
        64,
        dtype=torch.long,
    )

    labels = torch.tensor(
        [0, 1, 0, 1],
        dtype=torch.long,
    )

    logits = model(
        input_ids,
        attention_mask,
    )

    criterion = torch.nn.CrossEntropyLoss()

    loss = criterion(
        logits,
        labels,
    )

    loss.backward()

    gradient = model.embedding.weight.grad

    assert gradient is not None

    assert torch.isfinite(gradient).all()


def test_classifier_with_real_dataset():

    tokenizer = IntentTokenizer()

    train_texts = load_training_texts(
        DATA_PATH
    )

    tokenizer.build_vocab(
        train_texts
    )

    dataset = IntentDataset(
        file_path=DATA_PATH,
        tokenizer=tokenizer,
        max_length=64,
    )

    batch_size = min(
        4,
        len(dataset),
    )

    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    batch = next(iter(dataloader))

    model = create_test_model(
        vocab_size=tokenizer.vocab_size,
    )

    logits = model(
        batch["input_ids"],
        batch["attention_mask"],
    )

    assert logits.shape == (
        batch_size,
        2,
    )

    assert torch.isfinite(logits).all()

    criterion = torch.nn.CrossEntropyLoss()

    loss = criterion(
        logits,
        batch["label"],
    )

    assert torch.isfinite(loss)

    loss.backward()