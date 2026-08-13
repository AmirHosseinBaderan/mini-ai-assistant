import pytest

from application.product_search import (
    Product,
    ProductSearchEngine,
    ProductSearchService,
    ParserRegistry,
    SiteConfig,
)


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


class FakeFetcher:

    async def fetch(
        self,
        url: str,
    ) -> str:

        return "<html>products</html>"


class FakeService:

    def __init__(
        self,
        fetcher,
        parser,
    ):
        self.fetcher = fetcher
        self.parser = parser
        self.calls = []

    async def search(
        self,
        url: str,
    ) -> list[Product]:

        self.calls.append(url)

        return self.parser.parse(
            "<html>products</html>"
        )


@pytest.mark.anyio
async def test_search():

    parser = FakeParser()

    registry = ParserRegistry()

    registry.register(
        "torob",
        parser,
    )

    fetcher = FakeFetcher()

    service = ProductSearchService(
        fetcher=fetcher,
        parser=parser,
    )

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