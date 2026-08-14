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

        base_url = self._get_base_url()

        return [
            {
                "name": product.name,
                "price": product.price,
                "url": self._normalize_url(
                    base_url,
                    product.url,
                ),
                "source": product.source,
            }
            for product in products[:take]
        ]

    def _get_base_url(self) -> str:
        """Extract base URL from the first site's search URL."""
        sites = getattr(self.engine, "sites", None)
        if sites:
            search_url = sites[0].search_url
            # Extract scheme + netloc from search_url
            # e.g., "https://torob.com/search/?query={query}" -> "https://torob.com"
            parts = search_url.split("/")
            return f"{parts[0]}//{parts[2]}"
        return "https://torob.com"

    def _normalize_url(
        self,
        base_url: str,
        url: str,
    ) -> str:
        """Normalize URL to absolute, decoded form."""
        if not url:
            return url

        # Join base URL with relative path
        absolute_url = urljoin(base_url, url)

        # Decode percent-encoded characters for readability
        decoded_url = unquote(absolute_url)

        return decoded_url