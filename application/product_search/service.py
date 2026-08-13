from application.product_search.config import SiteConfigLoader
from application.product_search.fetcher import HttpxFetcher


class ProductSearchService:

    def __init__(
        self,
        config_loader: SiteConfigLoader,
        fetcher: HttpxFetcher,
    ):
        self.config_loader = config_loader
        self.fetcher = fetcher

    async def search(
        self,
        query: str,
    ):

        if not query.strip():
            raise ValueError(
                "query is required"
            )

        sites = self.config_loader.load()

        results = []

        for site in sites:

            url = site.search_url.format(
                query=query,
            )

            html = await self.fetcher.fetch(
                url,
            )

            results.append(
                {
                    "site": site.name,
                    "url": url,
                    "html": html,
                }
            )

        return results