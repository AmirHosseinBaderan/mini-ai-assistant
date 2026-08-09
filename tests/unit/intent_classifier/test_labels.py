from intent_classifier.labels import (
    IntentLabel,
    label_to_id,
    id_to_label,
)


def test_labels():
    assert IntentLabel.CHAT == 0
    assert IntentLabel.RAG == 1


def test_label_to_id():
    assert label_to_id("chat") == 0
    assert label_to_id("rag") == 1


def test_id_to_label():
    assert id_to_label(0) == "chat"
    assert id_to_label(1) == "rag"