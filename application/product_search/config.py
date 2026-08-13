import json
from pathlib import Path

from application.product_search.models import (
    SiteConfig,
)


class SiteConfigLoader:

    def __init__(
        self,
        path: str | Path,
    ):
        self.path = Path(path)

    def load(self) -> list[SiteConfig]:

        with self.path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        return [
            SiteConfig(
                name=site["name"],
                search_url=site["search_url"],
            )
            for site in data["sites"]
        ]