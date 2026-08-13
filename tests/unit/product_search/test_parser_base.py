import pytest

from application.product_search.parsers.base import ParserBase


def test_parser_base_requires_parse():

    with pytest.raises(
        TypeError
    ):
        ParserBase()