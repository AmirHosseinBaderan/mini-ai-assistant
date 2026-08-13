import json

import pytest

from application.product_search.config import (
    SiteConfigLoader,
)


def test_load_sites(tmp_path):

    config_path = tmp_path / "sites.json"

    config_path.write_text(
        json.dumps(
            {
                "sites": [
                    {
                        "name": "example",
                        "search_url": (
                            "https://example.com/search?q={query}"
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    loader = SiteConfigLoader(config_path)

    sites = loader.load()

    assert len(sites) == 1

    assert sites[0].name == "example"

    assert (
        sites[0].search_url
        == "https://example.com/search?q={query}"
    )


def test_load_empty_sites(tmp_path):

    config_path = tmp_path / "sites.json"

    config_path.write_text(
        json.dumps(
            {"sites": []}
        ),
        encoding="utf-8",
    )

    loader = SiteConfigLoader(config_path)

    assert loader.load() == []


def test_invalid_search_url(tmp_path):

    config_path = tmp_path / "sites.json"

    config_path.write_text(
        json.dumps(
            {
                "sites": [
                    {
                        "name": "example",
                        "search_url": (
                            "https://example.com/search"
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    loader = SiteConfigLoader(config_path)

    with pytest.raises(ValueError):
        loader.load()