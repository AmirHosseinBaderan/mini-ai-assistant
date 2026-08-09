from dataclasses import dataclass
from pathlib import Path

import torch

from application.utils.logger import get_logger

from .checkpoint import CheckpointManager
from .config import IntentConfig
from .labels import id_to_label
from .model import IntentClassifier
from .tokenizer import IntentTokenizer


logger = get_logger("intent.predictor")


@dataclass
class ClassificationResult:
    label: str
    confidence: float


class IntentPredictor:

    def __init__(
        self,
        tokenizer: IntentTokenizer,
        config: IntentConfig | None = None,
        checkpoint_dir: str | Path | None = None,
        use_best: bool = True,
    ):
        self.device = config.device if config else torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.tokenizer = tokenizer
        self.config = config or IntentConfig()

        checkpoint_manager = CheckpointManager(
            checkpoint_dir=Path(checkpoint_dir)
            if checkpoint_dir is not None
            else self.config.checkpoint_dir,
        )

        if use_best:
            checkpoint_path = (
                checkpoint_manager.best_checkpoint_path()
            )
        else:
            checkpoint_path = (
                checkpoint_manager.last_checkpoint_path()
            )

        checkpoint = checkpoint_manager.load(checkpoint_path)

        self.model = IntentClassifier(
            vocab_size=tokenizer.vocab_size,
            num_classes=self.config.num_classes,
            embed_dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            ff_dim=self.config.ff_dim,
            max_length=self.config.max_length,
            dropout=self.config.dropout,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)

        self.model.eval()

        logger.info(
            "Predictor initialized with checkpoint: %s",
            checkpoint_path,
        )

    @torch.no_grad()
    def predict(
        self,
        text: str,
    ) -> ClassificationResult:

        token_ids = self.tokenizer.encode(
            text,
            max_length=self.config.max_length,
        )

        attention_mask = [
            0 if token_id == self.tokenizer.pad_id else 1
            for token_id in token_ids
        ]

        input_ids = torch.tensor(
            token_ids,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)

        attention_mask = torch.tensor(
            attention_mask,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)

        logits = self.model(
            input_ids,
            attention_mask,
        )

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        class_id = torch.argmax(
            probabilities,
            dim=-1,
        ).item()

        confidence = probabilities[
            0,
            class_id,
        ].item()

        label = id_to_label(class_id)

        logger.debug(
            "Predicted '%s' -> %s (%.2f%%)",
            text,
            label,
            confidence * 100,
        )

        return ClassificationResult(
            label=label,
            confidence=confidence,
        )
