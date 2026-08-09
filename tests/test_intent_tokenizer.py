from intent_classifier.tokenizer import IntentTokenizer


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