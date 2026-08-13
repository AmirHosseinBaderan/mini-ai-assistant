from application.product_search.models import Product


class ProductSearchService:

    def __init__(
        self,
        fetcher,
    ):
        self.fetcher = fetcher

    async def search(
        self,
        url: str,
        parser,
    ) -> list[Product]:

        html = await self.fetcher.fetch(
            url,
        )

        return parser.parse(
            html,
        )