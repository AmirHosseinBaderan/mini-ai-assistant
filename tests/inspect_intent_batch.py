import json
from pathlib import Path

from intent_classifier.dataset import (
    IntentDataset,
    create_dataloader,
)
from intent_classifier.tokenizer import IntentTokenizer


DATA_PATH = Path(
    "data/intent/train.jsonl"
)


def load_texts(path: Path) -> list[str]:

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


tokenizer = IntentTokenizer()

train_texts = load_texts(DATA_PATH)

tokenizer.build_vocab(
    train_texts
)

dataset = IntentDataset(
    file_path=DATA_PATH,
    tokenizer=tokenizer,
    max_length=64,
)

dataloader = create_dataloader(
    dataset,
    batch_size=4,
    shuffle=False,
)

batch = next(iter(dataloader))

print("Vocabulary size:")
print(tokenizer.vocab_size)

print()

print("input_ids:")
print(batch["input_ids"])

print()

print("attention_mask:")
print(batch["attention_mask"])

print()

print("labels:")
print(batch["label"])

print()

print("Shapes:")
print(
    "input_ids:",
    batch["input_ids"].shape,
)

print(
    "attention_mask:",
    batch["attention_mask"].shape,
)

print(
    "labels:",
    batch["label"].shape,
)