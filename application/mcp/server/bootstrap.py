from application.mcp import ProductSearchTool
from application.product_search import (
    HttpxFetcher,
    ParserRegistry,
    ProductSearchEngine,
    ProductSearchService,
    SiteConfigLoader,
)
from application.product_search.parsers.torob import (
    TorobParser,
)

from application.mcp.server.server import (
    mcp,
    register_product_search,
)


def create_product_search_tool() -> ProductSearchTool:

    loader = SiteConfigLoader(
        "resources/sites.json",
    )

    sites = loader.load()

    registry = ParserRegistry()

    registry.register(
        "torob",
        TorobParser(),
    )

    fetcher = HttpxFetcher()

    service = ProductSearchService(
        fetcher=fetcher,
    )

    engine = ProductSearchEngine(
        sites=sites,
        parser_registry=registry,
        search_service=service,
    )

    return ProductSearchTool(
        engine=engine,
    )


def create_server():

    tool = create_product_search_tool()

    register_product_search(
        tool,
    )

    return mcp