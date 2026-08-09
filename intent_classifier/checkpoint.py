from pathlib import Path

import torch


class CheckpointManager:

    def __init__(
        self,
        checkpoint_dir: str | Path,
        monitor: str = "validation_f1",
        mode: str = "max",
        top_k: int = 1,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.monitor = monitor
        self.mode = mode
        self.top_k = top_k

        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")

        self.best_value = (
            float("-inf") if mode == "max" else float("inf")
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def is_better(
        self,
        current: float,
    ) -> bool:
        if self.mode == "max":
            return current > self.best_value

        return current < self.best_value

    def update_best(self, value: float) -> None:
        self.best_value = value

    def save(
        self,
        path: str | Path,
        epoch: int,
        model_state_dict: dict,
        optimizer_state_dict: dict,
        metrics: dict[str, float],
    ) -> None:
        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer_state_dict,
                "metrics": metrics,
            },
            path,
        )

    def save_last(
        self,
        epoch: int,
        model_state_dict: dict,
        optimizer_state_dict: dict,
        metrics: dict[str, float],
    ) -> Path:
        path = self.checkpoint_dir / "last.pt"
        self.save(
            path=path,
            epoch=epoch,
            model_state_dict=model_state_dict,
            optimizer_state_dict=optimizer_state_dict,
            metrics=metrics,
        )
        return path

    def save_best(
        self,
        epoch: int,
        model_state_dict: dict,
        optimizer_state_dict: dict,
        metrics: dict[str, float],
    ) -> Path:
        path = self.checkpoint_dir / "best.pt"
        self.save(
            path=path,
            epoch=epoch,
            model_state_dict=model_state_dict,
            optimizer_state_dict=optimizer_state_dict,
            metrics=metrics,
        )
        return path

    def load(
        self,
        path: str | Path,
    ) -> dict:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
