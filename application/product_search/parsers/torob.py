from bs4 import BeautifulSoup

from application.product_search.models import Product
from application.product_search.parsers.base import ParserBase


class TorobParser(ParserBase):

    def parse(
        self,
        html: str,
    ) -> list[Product]:

        soup = BeautifulSoup(
            html,
            "html.parser",
        )

        products = []

        for card in soup.select(
            '[data-testid="product-card"]'
        ):

            link = card.find_parent("a")
            name = card.find("h2")

            price = card.find(
                "div",
                class_=lambda value: (
                    value
                    and "product-price-text"
                    in value
                ),
            )

            if not link or not name:
                continue

            products.append(
                Product(
                    name=name.get_text(
                        " ",
                        strip=True,
                    ),
                    price=(
                        price.get_text(
                            " ",
                            strip=True,
                        )
                        if price
                        else None
                    ),
                    url=link.get("href", ""),
                    source="torob",
                )
            )

        return products