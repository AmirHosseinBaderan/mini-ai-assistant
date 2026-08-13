from application.product_search import ParserBase


class ParserRegistry:

    def __init__(self):

        self._parsers: dict[
            str,
            ParserBase,
        ] = {}

    def register(
        self,
        name: str,
        parser: ParserBase,
    ) -> None:

        if name in self._parsers:
            raise ValueError(
                f"Parser already registered: {name}"
            )

        self._parsers[name] = parser

    def get(
        self,
        name: str,
    ) -> ParserBase:

        try:
            return self._parsers[name]

        except KeyError:
            raise KeyError(
                f"Parser not found: {name}"
            )

    def all(self) -> list[ParserBase]:

        return list(
            self._parsers.values()
        )