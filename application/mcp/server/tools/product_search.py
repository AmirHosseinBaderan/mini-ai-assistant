from typing import Any


async def search_products(
    query: str,
) -> list[dict[str, Any]]:
    """
    Search products across configured websites.

    Args:
        query: Product name or search query.

    Returns:
        A list of products with name, price, currency,
        source and url.
    """

    if not query.strip():
        raise ValueError(
            "query is required"
        )

    # Product search service will be connected here.
    raise NotImplementedError