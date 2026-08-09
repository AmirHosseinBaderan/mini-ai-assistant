from pathlib import Path

from torch.utils.data import DataLoader

from intent_classifier.checkpoint import CheckpointManager
from intent_classifier.config import IntentConfig
from intent_classifier.dataset import IntentDataset
from intent_classifier.early_stopping import EarlyStopping
from intent_classifier.model import IntentClassifier
from intent_classifier.tokenizer import IntentTokenizer
from intent_classifier.trainer import IntentTrainer


TRAIN_PATH = Path(
    "data/intent/train.jsonl"
)

VALIDATION_PATH = Path(
    "data/intent/validation.jsonl"
)

CHECKPOINT_PATH = Path(
    "checkpoints/intent"
)


def main():

    config = IntentConfig()

    tokenizer = IntentTokenizer()

    tokenizer.build_vocab_from_file(
        TRAIN_PATH
    )

    train_dataset = IntentDataset(
        file_path=TRAIN_PATH,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )

    validation_dataset = IntentDataset(
        file_path=VALIDATION_PATH,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = IntentClassifier(
        vocab_size=tokenizer.vocab_size,
        num_classes=config.num_classes,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ff_dim=config.ff_dim,
        max_length=config.max_length,
        dropout=config.dropout,
    )

    trainer = IntentTrainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=config.device,
    )

    early_stopping = EarlyStopping(
        monitor="validation_f1",
        mode="max",
        patience=5,
        min_delta=0.001,
    )

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=CHECKPOINT_PATH,
        monitor="validation_f1",
        mode="max",
    )

    history, best_epoch = trainer.fit(
        epochs=config.epochs,
        checkpoint_path=CHECKPOINT_PATH / "best.pt",
        early_stopping=early_stopping,
    )

    best_checkpoint = checkpoint_manager.load(
        CHECKPOINT_PATH / "best.pt"
    )

    print()
    print("Training completed.")
    print(
        f"Best Epoch: {best_epoch}"
    )
    print(
        f"Validation F1: "
        f"{best_checkpoint['metrics']['validation_f1']:.4f}"
    )
    print(
        f"Best Checkpoint: {CHECKPOINT_PATH / 'best.pt'}"
    )
    print(
        f"Last Checkpoint: {CHECKPOINT_PATH / 'last.pt'}"
    )


if __name__ == "__main__":
    main()
