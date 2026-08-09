from pathlib import Path

from torch.utils.data import DataLoader

from application.utils.logger import get_logger, setup_file_logger

from intent_classifier.checkpoint import CheckpointManager
from intent_classifier.config import IntentConfig
from intent_classifier.dataset import IntentDataset
from intent_classifier.early_stopping import EarlyStopping
from intent_classifier.model import IntentClassifier
from intent_classifier.tokenizer import IntentTokenizer
from intent_classifier.trainer import IntentTrainer


logger = get_logger("intent.train")

TRAIN_PATH = Path(
    "data/intent/train.jsonl"
)

VALIDATION_PATH = Path(
    "data/intent/validation.jsonl"
)


def main():

    config = IntentConfig()

    setup_file_logger(
        logger,
        log_dir=config.tensorboard_log_dir,
        filename="training.log",
    )

    logger.info("Starting intent classifier training")
    logger.info("Config: %s", config)

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
        config=config,
    )

    early_stopping = EarlyStopping(
        monitor="validation_f1",
        mode="max",
        patience=5,
        min_delta=0.001,
    )

    history, best_epoch = trainer.fit(
        epochs=config.epochs,
        checkpoint_dir=config.checkpoint_dir,
        early_stopping=early_stopping,
        log_dir=config.tensorboard_log_dir,
    )

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config.checkpoint_dir,
        monitor="validation_f1",
        mode="max",
    )

    best_checkpoint = checkpoint_manager.load(
        config.checkpoint_dir / "best.pt"
    )

    logger.info("Training completed.")
    logger.info("Best Epoch: %d", best_epoch)
    logger.info(
        "Validation F1: %.4f",
        best_checkpoint["metrics"]["validation_f1"],
    )
    logger.info(
        "Best Checkpoint: %s",
        config.checkpoint_dir / "best.pt",
    )
    logger.info(
        "Last Checkpoint: %s",
        config.checkpoint_dir / "last.pt",
    )


if __name__ == "__main__":
    main()
