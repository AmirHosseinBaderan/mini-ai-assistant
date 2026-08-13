import httpx


class HttpxFetcher:

    def __init__(
        self,
        timeout: float = 10.0,
    ):
        self.timeout = timeout

    async def fetch(
        self,
        url: str,
    ) -> str:

        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
        ) as client:

            response = await client.get(url)

            response.raise_for_status()

            return response.text