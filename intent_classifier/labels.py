from enum import IntEnum


class IntentLabel(IntEnum):
    CHAT = 0
    RAG = 1


LABEL_TO_ID = {
    "chat": IntentLabel.CHAT,
    "rag": IntentLabel.RAG,
}


ID_TO_LABEL = {
    IntentLabel.CHAT: "chat",
    IntentLabel.RAG: "rag",
}


def label_to_id(label: str) -> int:
    label = label.lower().strip()

    if label not in LABEL_TO_ID:
        raise ValueError(f"Unknown intent label: {label}")

    return int(LABEL_TO_ID[label])


def id_to_label(label_id: int) -> str:
    try:
        return ID_TO_LABEL[IntentLabel(label_id)]
    except ValueError:
        raise ValueError(f"Unknown intent id: {label_id}")