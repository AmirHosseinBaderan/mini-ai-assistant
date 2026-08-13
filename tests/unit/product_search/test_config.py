import json

from application.product_search import (
    SiteConfigLoader,
)


def test_load_sites(tmp_path):

    path = tmp_path / "sites.json"

    path.write_text(
        json.dumps(
            {
                "sites": [
                    {
                        "name": "torob",
                        "search_url": (
                            "https://torob.com/search/?query={query}"
                        ),
                    },
                    {
                        "name": "digikala",
                        "search_url": (
                            "https://digikala.com/search/?q={query}"
                        ),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    loader = SiteConfigLoader(
        path,
    )

    sites = loader.load()

    assert len(sites) == 2

    assert sites[0].name == "torob"
    assert sites[0].search_url == (
        "https://torob.com/search/?query={query}"
    )

    assert sites[1].name == "digikala"
    assert sites[1].search_url == (
        "https://digikala.com/search/?q={query}"
    )