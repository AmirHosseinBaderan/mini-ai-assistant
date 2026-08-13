import pytest

from application.product_search.parsers.registry import (
    ParserRegistry,
)


class FakeParser:

    def parse(
        self,
        html: str,
    ):
        return []


def test_register_parser():

    registry = ParserRegistry()

    parser = FakeParser()

    registry.register(
        "torob",
        parser,
    )

    assert registry.get(
        "torob"
    ) is parser


def test_list_parsers():

    registry = ParserRegistry()

    parser = FakeParser()

    registry.register(
        "torob",
        parser,
    )

    assert registry.all() == [
        parser
    ]


def test_duplicate_parser():

    registry = ParserRegistry()

    registry.register(
        "torob",
        FakeParser(),
    )

    with pytest.raises(ValueError):

        registry.register(
            "torob",
            FakeParser(),
        )


def test_unknown_parser():

    registry = ParserRegistry()

    with pytest.raises(KeyError):

        registry.get(
            "unknown",
        )