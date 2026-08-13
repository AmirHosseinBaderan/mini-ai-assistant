from typing import Any

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
    ) -> list[dict[str, Any]]:

        products = await self.engine.search(
            query,
        )

        return [
            {
                "name": product.name,
                "price": product.price,
                "url": product.url,
                "source": product.source,
            }
            for product in products
        ]