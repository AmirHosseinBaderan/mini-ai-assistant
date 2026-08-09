from pathlib import Path

from intent_classifier.dataset import (
    IntentDataset,
    create_dataloader,
)
from intent_classifier.tokenizer import IntentTokenizer
import torch

DATA_DIR = Path("data/intent")


def load_training_texts(
    path: Path,
) -> list[str]:

    import json

    texts = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if not line.strip():
                continue

            record = json.loads(line)

            texts.append(
                record["text"]
            )

    return texts


def test_dataset():

    train_path = DATA_DIR / "train.jsonl"

    tokenizer = IntentTokenizer()

    train_texts = load_training_texts(
        train_path
    )

    tokenizer.build_vocab(
        train_texts
    )

    dataset = IntentDataset(
        file_path=train_path,
        tokenizer=tokenizer,
        max_length=64,
    )

    assert len(dataset) > 0

    sample = dataset[0]

    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "label" in sample

    assert sample["input_ids"].shape == (64,)
    assert sample["attention_mask"].shape == (64,)
    assert sample["label"].shape == ()

    assert sample["label"].item() in (0, 1)


def test_persian_sample():

    train_path = DATA_DIR / "train.jsonl"

    tokenizer = IntentTokenizer()

    train_texts = load_training_texts(
        train_path
    )

    tokenizer.build_vocab(
        train_texts
    )

    dataset = IntentDataset(
        file_path=train_path,
        tokenizer=tokenizer,
        max_length=64,
    )

    found_persian = False

    for index in range(len(dataset)):

        text = dataset.samples[index]["text"]

        if any(
            "\u0600" <= char <= "\u06FF"
            for char in text
        ):
            found_persian = True

            sample = dataset[index]

            assert sample["input_ids"].shape == (64,)
            assert sample["attention_mask"].sum().item() > 0

            break

    assert found_persian

def test_persian_tokenization():

    tokenizer = IntentTokenizer()

    tokens = tokenizer.tokenize(
        "نحوه کار داینامو‌دی‌بی را توضیح بده"
    )

    assert len(tokens) > 0
    assert "نحوه" in tokens
    assert "کار" in tokens
    assert "را" in tokens
    assert "توضیح" in tokens
    assert "بده" in tokens
    
def test_english_tokenization():

    tokenizer = IntentTokenizer()

    tokens = tokenizer.tokenize(
        "What is Python?"
    )

    assert tokens == [
        "what",
        "is",
        "python",
        "?",
    ]
    
def test_dataloader():

    train_path = DATA_DIR / "train.jsonl"

    tokenizer = IntentTokenizer()

    train_texts = load_training_texts(
        train_path
    )

    tokenizer.build_vocab(
        train_texts
    )

    dataset = IntentDataset(
        file_path=train_path,
        tokenizer=tokenizer,
        max_length=64,
    )

    dataloader = create_dataloader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    batch = next(iter(dataloader))

    assert batch["input_ids"].shape == (
        4,
        64,
    )

    assert batch["attention_mask"].shape == (
        4,
        64,
    )

    assert batch["label"].shape == (
        4,
    )

    assert batch["input_ids"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.long
    assert batch["label"].dtype == torch.long