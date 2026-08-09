import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .labels import label_to_id
from .tokenizer import IntentTokenizer
from torch.utils.data import DataLoader

class IntentDataset(Dataset):

    def __init__(
        self,
        file_path: str | Path,
        tokenizer: IntentTokenizer,
        max_length: int = 64,
    ):
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.samples = self._load()

    def _load(self) -> list[dict]:
        samples = []

        with self.file_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(file, start=1):

                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at line {line_number}"
                    ) from exc

                if "text" not in record:
                    raise ValueError(
                        f"Missing 'text' at line {line_number}"
                    )

                if "label" not in record:
                    raise ValueError(
                        f"Missing 'label' at line {line_number}"
                    )

                samples.append({
                    "text": record["text"],
                    "label": label_to_id(record["label"]),
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:

        sample = self.samples[index]

        input_ids = self.tokenizer.encode(
            sample["text"],
            max_length=self.max_length,
        )

        attention_mask = [
            0 if token_id == self.tokenizer.pad_id else 1
            for token_id in input_ids
        ]

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
            ),
            "label": torch.tensor(
                sample["label"],
                dtype=torch.long,
            ),
        }
        

def create_dataloader(
        dataset: IntentDataset,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> DataLoader:

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )