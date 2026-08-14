from typing import Any
from urllib.parse import unquote, urljoin

from application.product_search.engine import ProductSearchEngine


class ProductSearchTool:

    def __init__(
        self,
        engine: ProductSearchEngine,
    ):
        self.engine = engine

    async def execute(
        self,
        query: str,
        take: int = 10,
    ) -> list[dict[str, Any]]:

        products = await self.engine.search(
            query,
        )

        return [
            {
                "name": product.name,
                "price": product.price,
                "url": self._normalize_url(
                    product.source,
                    product.url,
                ),
                "source": product.source,
            }
            for product in products[:take]
        ]

    def _get_base_url_for_source(
        self,
        source: str,
    ) -> str:
        """Extract base URL from the site config matching the given source."""
        sites = getattr(self.engine, "sites", None)
        if sites:
            for site in sites:
                if site.name == source:
                    # Extract scheme + netloc from search_url
                    # e.g., "https://torob.com/search/?query={query}" -> "https://torob.com"
                    parts = site.search_url.split("/")
                    return f"{parts[0]}//{parts[2]}"
        return ""

    def _normalize_url(
        self,
        source: str,
        url: str,
    ) -> str:
        """Normalize URL to absolute, decoded form using the source's base URL."""
        if not url:
            return url

        base_url = self._get_base_url_for_source(source)

        # Join base URL with relative path
        absolute_url = urljoin(base_url, url)

        # Decode percent-encoded characters for readability
        decoded_url = unquote(absolute_url)

        return decoded_url