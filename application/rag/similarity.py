import math


def cosine_similarity(
    first: list[float],
    second: list[float],
) -> float:

    if len(first) != len(second):
        raise ValueError(
            "Vectors must have the same dimension"
        )

    if not first:
        raise ValueError(
            "Vectors cannot be empty"
        )

    dot_product = sum(
        a * b
        for a, b in zip(first, second)
    )

    first_norm = math.sqrt(
        sum(a * a for a in first)
    )

    second_norm = math.sqrt(
        sum(b * b for b in second)
    )

    if first_norm == 0 or second_norm == 0:
        raise ValueError(
            "Zero vectors are not supported"
        )

    return dot_product / (
        first_norm * second_norm
    )