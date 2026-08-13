from dataclasses import dataclass


@dataclass(frozen=True)
class SiteConfig:

    name: str
    search_url: str
    parser: str


@dataclass(frozen=True)
class Product:

    name: str
    price: str | None
    url: str
    source: str