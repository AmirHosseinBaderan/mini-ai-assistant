import pytest

from application.product_search.service import (
    ProductSearchService,
)

class FakeFetcher:

    def __init__(self):
        self.urls = []

    async def fetch(
        self,
        url: str,
    ) -> str:

        self.urls.append(url)

        return "<html>test</html>"


class FakeParser:

    def __init__(self):
        self.html = None

    def parse(
        self,
        html: str,
    ):

        self.html = html

        return [
            {
                "name": "iPhone 16",
                "price": "160000000",
            }
        ]


@pytest.mark.anyio
async def test_search():

    fetcher = FakeFetcher()
    parser = FakeParser()

    service = ProductSearchService(
        fetcher=fetcher,
    )

    result = await service.search(
        "https://example.com/search",
        parser=parser,
    )

    assert fetcher.urls == [
        "https://example.com/search"
    ]

    assert parser.html == (
        "<html>test</html>"
    )

    assert result == [
        {
            "name": "iPhone 16",
            "price": "160000000",
        }
    ]