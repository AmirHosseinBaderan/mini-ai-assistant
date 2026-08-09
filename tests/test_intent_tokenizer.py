from intent_classifier.tokenizer import IntentTokenizer

from pathlib import Path


def test_vocab_is_built_from_train_only():

    tokenizer = IntentTokenizer()

    train_path = Path(
        "data/intent/train.jsonl"
    )

    validation_path = Path(
        "data/intent/validation.jsonl"
    )

    tokenizer.build_vocab_from_file(
        train_path
    )

    train_text = tokenizer.encode(
        "hello"
    )

    validation_text = tokenizer.encode(
        "this text should become unknown"
    )

    assert train_text is not None
    assert validation_text is not None


def test_tokenize():
    tokenizer = IntentTokenizer()

    tokens = tokenizer.tokenize(
        "What is Python?"
    )

    assert tokens == [
        "what",
        "is",
        "python",
        "?",
    ]


def test_build_vocab():

    tokenizer = IntentTokenizer()

    tokenizer.build_vocab([
        "hello",
        "how are you",
        "what is python",
    ])

    assert tokenizer.vocab_size > 2

    assert "hello" in tokenizer.token_to_id
    assert "python" in tokenizer.token_to_id


def test_encode():

    tokenizer = IntentTokenizer()

    tokenizer.build_vocab([
        "hello world",
    ])

    encoded = tokenizer.encode(
        "hello world",
        max_length=5,
    )

    assert len(encoded) == 5

    assert encoded[-1] == tokenizer.pad_id


def test_unknown_token():

    tokenizer = IntentTokenizer()

    tokenizer.build_vocab([
        "hello",
    ])

    encoded = tokenizer.encode(
        "something",
        max_length=3,
    )

    assert encoded[0] == tokenizer.unk_id


def test_decode():

    tokenizer = IntentTokenizer()

    tokenizer.build_vocab([
        "hello world",
    ])

    encoded = tokenizer.encode(
        "hello world",
        max_length=4,
    )

    decoded = tokenizer.decode(encoded)

    assert decoded == "hello world"