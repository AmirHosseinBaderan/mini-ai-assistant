from dataclasses import dataclass, field


@dataclass
class EarlyStopping:
    monitor: str = "validation_f1"
    mode: str = "max"
    patience: int = 5
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")

        self.best_value = (
            float("-inf") if self.mode == "max" else float("inf")
        )

        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False

    def is_better(
        self,
        current: float,
    ) -> bool:
        if self.mode == "max":
            return current > self.best_value + self.min_delta

        return current < self.best_value - self.min_delta

    def step(
        self,
        current: float,
        epoch: int,
    ) -> bool:
        if self.is_better(current):
            self.best_value = current
            self.wait = 0
        else:
            self.wait += 1

            if self.wait >= self.patience:
                self.stop_training = True
                self.stopped_epoch = epoch

        return self.stop_training
