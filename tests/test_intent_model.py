import torch

from intent_classifier.model import (
    IntentTransformerEncoder,
)

import json
from pathlib import Path

from intent_classifier.dataset import (
    IntentDataset,
    create_dataloader,
)
from intent_classifier.tokenizer import (
    IntentTokenizer,
)


DATA_PATH = Path(
    "data/intent/train.jsonl"
)


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

            texts.append(
                record["text"]
            )

    return texts

def test_model_shape():

    model = IntentTransformerEncoder(
        vocab_size=1017,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        max_length=64,
        dropout=0.1,
    )

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

    output = model(
        input_ids,
        attention_mask,
    )

    assert output.shape == (
        4,
        64,
        128,
    )
    
def test_model_with_padding():

    model = IntentTransformerEncoder(
        vocab_size=1017,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        max_length=64,
    )

    input_ids = torch.tensor([
        [2, 0, 0, 0, 0, 0, 0, 0],
        [3, 4, 5, 6, 7, 0, 0, 0],
    ])

    attention_mask = torch.tensor([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
    ])

    output = model(
        input_ids,
        attention_mask,
    )

    assert output.shape == (
        2,
        8,
        128,
    )

    assert torch.isfinite(output).all()
    
def test_model_with_real_dataset():

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

    model = IntentTransformerEncoder(
        vocab_size=tokenizer.vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        max_length=64,
    )

    output = model(
        batch["input_ids"],
        batch["attention_mask"],
    )

    assert output.shape == (
        batch_size,
        64,
        128,
    )

    assert torch.isfinite(
        output
    ).all()
    
def test_model_backward():

    model = IntentTransformerEncoder(
        vocab_size=1017,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        max_length=64,
    )

    input_ids = torch.randint(
        0,
        1017,
        (2, 64),
    )

    attention_mask = torch.ones(
        2,
        64,
        dtype=torch.long,
    )

    output = model(
        input_ids,
        attention_mask,
    )

    loss = output.mean()

    loss.backward()

    assert model.embedding.weight.grad is not None

    assert torch.isfinite(
        model.embedding.weight.grad
    ).all()