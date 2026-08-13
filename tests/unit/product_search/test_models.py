from application.product_search.models import Product


def test_product():

    product = Product(
        name="Laptop",
        price="$999",
        url="https://example.com/laptop",
        source="example",
    )

    assert product.name == "Laptop"
    assert product.price == "$999"
    assert product.url == "https://example.com/laptop"
    assert product.source == "example"