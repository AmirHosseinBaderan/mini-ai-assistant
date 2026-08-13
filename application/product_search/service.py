from application.product_search.models import Product


class ProductSearchService:

    def __init__(
        self,
        fetcher,
        parser,
    ):
        self.fetcher = fetcher
        self.parser = parser

    async def search(
        self,
        url: str,
    ) -> list[Product]:

        html = await self.fetcher.fetch(
            url,
        )

        return self.parser.parse(
            html,
        )