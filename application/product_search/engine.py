from application.product_search.models import Product, SiteConfig
from application.product_search.parsers.registry import ParserRegistry
from application.product_search.service import ProductSearchService


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

            url = site.search_url.format(
                query=query,
            )

            results = await self.search_service.search(
                url=url,
                parser=parser,
            )

            products.extend(
                results,
            )

        return products