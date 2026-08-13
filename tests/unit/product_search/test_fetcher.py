import httpx
import pytest

from application.product_search.fetcher import (
    HttpxFetcher,
)


@pytest.mark.anyio
async def test_fetch():

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        return httpx.Response(
            200,
            text="<html>Hello</html>",
        )

    transport = httpx.MockTransport(handler)

    fetcher = HttpxFetcher()

    async with httpx.AsyncClient(
        transport=transport,
    ):
        pass