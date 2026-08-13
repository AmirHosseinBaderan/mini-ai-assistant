from application.product_search import (
    Product,
    SiteConfig,
    ProductSearchService,
    ParserRegistry,
)


class ProductSearchEngine:

    def __init__(
        self,
        sites: list[SiteConfig],
        parser_registry: ParserRegistry,
        search_service: ProductSearchService,
    ):
        self.sites = sites
        self.parser_registry = parser_registry
        self.search_service = search_service

    async def search(
        self,
        query: str,
    ) -> list[Product]:

        products = []

        for site in self.sites:

            parser = self.parser_registry.get(
                site.parser,
            )

            service = ProductSearchService(
                fetcher=self.search_service.fetcher,
                parser=parser,
            )

            url = site.search_url.format(
                query=query,
            )

            results = await service.search(
                url,
            )

            products.extend(
                results,
            )

        return products