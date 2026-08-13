from abc import ABC, abstractmethod

from application.product_search.models import Product


class ParserBase(ABC):

    @abstractmethod
    def parse(
        self,
        html: str,
    ) -> list[Product]:
        pass