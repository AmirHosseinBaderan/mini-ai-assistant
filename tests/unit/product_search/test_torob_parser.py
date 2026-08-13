from application.product_search import (
    TorobParser,
)


def test_parse_product():

    html = """
    <a href="/p/test-product">

        <div data-testid="product-card">

            <h2>
                گوشی آیفون 16
            </h2>

            <div
                class="ProductCard_desktop_product-price-text__test"
            >
                از ۱۶۰٫۰۰۰٫۰۰۰ تومان
            </div>

        </div>

    </a>
    """

    parser = TorobParser()

    products = parser.parse(
        html,
    )

    assert len(products) == 1

    product = products[0]

    assert product.name == "گوشی آیفون 16"

    assert (
        product.price
        == "از ۱۶۰٫۰۰۰٫۰۰۰ تومان"
    )

    assert product.url == "/p/test-product"

    assert product.source == "torob"