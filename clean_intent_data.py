import json
from pathlib import Path

TRAIN_PATH = Path("data/intent/train.jsonl")
VAL_PATH = Path("data/intent/validation.jsonl")


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_records(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    print("Loading records...")
    train_records = load_records(TRAIN_PATH)
    val_records = load_records(VAL_PATH)

    print(f"Train: {len(train_records)} records")
    print(f"Validation: {len(val_records)} records")

    # Deduplicate train by text, keep first occurrence
    train_by_text = {}
    for record in train_records:
        text = record.get("text", "")
        if text not in train_by_text:
            train_by_text[text] = record

    deduped_train = list(train_by_text.values())
    print(f"Train after dedup: {len(deduped_train)} records (removed {len(train_records) - len(deduped_train)} duplicates)")

    # Deduplicate validation by text, keep first occurrence
    val_by_text = {}
    for record in val_records:
        text = record.get("text", "")
        if text not in val_by_text:
            val_by_text[text] = record

    deduped_val = list(val_by_text.values())
    print(f"Validation after dedup: {len(deduped_val)} records (removed {len(val_records) - len(deduped_val)} duplicates)")

    # Remove validation items that exist in train
    train_texts = set(train_by_text.keys())
    cleaned_val = [r for r in deduped_val if r.get("text", "") not in train_texts]
    print(f"Validation after removing train overlap: {len(cleaned_val)} records (removed {len(deduped_val) - len(cleaned_val)} overlapping items)")

    # Save cleaned files
    save_records(TRAIN_PATH, deduped_train)
    save_records(VAL_PATH, cleaned_val)

    print("\nCleaned files saved:")
    print(f"  {TRAIN_PATH}: {len(deduped_train)} records")
    print(f"  {VAL_PATH}: {len(cleaned_val)} records")

    # Verify
    final_train = load_records(TRAIN_PATH)
    final_val = load_records(VAL_PATH)
    final_train_texts = {r.get("text", "") for r in final_train}
    final_val_texts = {r.get("text", "") for r in final_val}

    assert len(final_train) == len(final_train_texts), "Train still has duplicates!"
    assert len(final_val) == len(final_val_texts), "Validation still has duplicates!"
    assert not (final_train_texts & final_val_texts), "Train and validation still overlap!"
    print("\nVerification passed: all items are unique and no overlap exists.")


if __name__ == "__main__":
    main()
