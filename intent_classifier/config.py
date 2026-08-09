from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class IntentConfig:

    max_length: int = 64

    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.1

    num_classes: int = 2

    batch_size: int = 32

    learning_rate: float = 3e-4
    weight_decay: float = 1e-2

    epochs: int = 20

    checkpoint_dir: Path = field(
        default_factory=lambda: Path("checkpoints/intent")
    )

    tensorboard_log_dir: Path = field(
        default_factory=lambda: Path("logs/intent")
    )

    @property
    def device(self) -> torch.device:
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
