from unittest.mock import AsyncMock, Mock

import pytest

from application.product_search.models import SiteConfig
from application.product_search.service import (
    ProductSearchService,
)


@pytest.mark.anyio
async def test_search():

    config_loader = Mock()

    config_loader.load.return_value = [
        SiteConfig(
            name="example",
            search_url=(
                "https://example.com/search?q={query}"
            ),
        ),
    ]

    fetcher = Mock()

    fetcher.fetch = AsyncMock(
        return_value="<html>products</html>",
    )

    service = ProductSearchService(
        config_loader=config_loader,
        fetcher=fetcher,
    )

    result = await service.search(
        "laptop",
    )

    fetcher.fetch.assert_awaited_once_with(
        "https://example.com/search?q=laptop",
    )

    assert result == [
        {
            "site": "example",
            "url": (
                "https://example.com/search?q=laptop"
            ),
            "html": "<html>products</html>",
        }
    ]

@pytest.mark.anyio
async def test_search_requires_query():

    config_loader = Mock()
    fetcher = Mock()

    service = ProductSearchService(
        config_loader=config_loader,
        fetcher=fetcher,
    )

    with pytest.raises(ValueError):

        await service.search("")