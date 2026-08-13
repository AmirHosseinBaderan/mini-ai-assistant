import pytest

from application.product_search.models import Product, SiteConfig
from application.product_search.parsers.registry import ParserRegistry
from application.product_search.service import ProductSearchService
from application.product_search.engine import ProductSearchEngine


class FakeParser:

    def parse(
        self,
        html: str,
    ) -> list[Product]:

        return [
            Product(
                name="iPhone 16",
                price="160000000",
                url="https://example.com/product",
                source="torob",
            )
        ]


class FakeService:

    def __init__(self):

        self.calls = []

    async def search(
        self,
        url: str,
        parser,
    ) -> list[Product]:

        self.calls.append(
            {
                "url": url,
                "parser": parser,
            }
        )

        return [
            Product(
                name="iPhone 16",
                price="160000000",
                url="https://example.com/product",
                source="torob",
            )
        ]


@pytest.mark.anyio
async def test_search():

    parser = FakeParser()

    registry = ParserRegistry()

    registry.register(
        "torob",
        parser,
    )

    service = FakeService()

    engine = ProductSearchEngine(
        sites=[
            SiteConfig(
                name="torob",
                search_url=(
                    "https://torob.com/search/?query={query}"
                ),
                parser="torob",
            )
        ],
        parser_registry=registry,
        search_service=service,
    )

    result = await engine.search(
        "iphone 16",
    )

    assert result == [
        Product(
            name="iPhone 16",
            price="160000000",
            url="https://example.com/product",
            source="torob",
        )
    ]

    assert len(service.calls) == 1

    assert service.calls[0]["url"] == (
        "https://torob.com/search/?query=iphone 16"
    )

    assert service.calls[0]["parser"] is parser