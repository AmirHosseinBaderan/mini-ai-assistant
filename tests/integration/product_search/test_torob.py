import pytest

from application.product_search.fetcher import HttpxFetcher


@pytest.mark.anyio
async def test_fetch_torob():

    fetcher = HttpxFetcher()

    html = await fetcher.fetch(
        "https://torob.com/search/?query=iphone"
    )

    with open(
        "./tests/fixtures/torob_search.html",
        "w",
        encoding="utf-8",
    ) as file:

        file.write(html)

    assert html