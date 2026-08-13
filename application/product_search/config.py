import json
from pathlib import Path

from application.product_search.models import SiteConfig


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

        sites = data.get("sites")

        if not isinstance(sites, list):
            raise ValueError(
                "'sites' must be a list"
            )

        result = []

        for site in sites:

            if not isinstance(site, dict):
                raise ValueError(
                    "Each site must be an object"
                )

            name = site.get("name")
            search_url = site.get("search_url")

            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    "Site name is required"
                )

            if (
                not isinstance(search_url, str)
                or not search_url.strip()
            ):
                raise ValueError(
                    "Site search_url is required"
                )

            if "{query}" not in search_url:
                raise ValueError(
                    "search_url must contain {query}"
                )

            result.append(
                SiteConfig(
                    name=name,
                    search_url=search_url,
                )
            )

        return result