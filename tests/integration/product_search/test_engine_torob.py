from pathlib import Path

import pytest

from application.product_search import (
    HttpxFetcher,
    ParserRegistry,
    ProductSearchEngine,
    ProductSearchService,
    SiteConfigLoader,
)
from application.product_search.parsers.torob import (
    TorobParser,
)


@pytest.mark.anyio
async def test_search_torob():

    loader = SiteConfigLoader(
        Path("resources/sites.json"),
    )

    sites = loader.load()

    registry = ParserRegistry()

    registry.register(
        "torob",
        TorobParser(),
    )

    fetcher = HttpxFetcher()

    service = ProductSearchService(
        fetcher=fetcher,
    )

    engine = ProductSearchEngine(
        sites=sites,
        parser_registry=registry,
        search_service=service,
    )

    results = await engine.search(
        "iphone 16",
    )
    print(f'Products count {len(results)}')
    assert results

    for product in results:
        print(f'name : {product.name} / price : {product.price}')
        assert product.name
        assert product.url
        assert product.source == "torob"