import pytest

from src.calculator import calculator


@pytest.mark.parametrize(
    ("operator", "left", "right", "expected"),
    [
        ("add", 2, 3, 5),
        ("subtract", 7, 2, 5),
        ("multiply", 6, 4, 24),
        ("divide", 9, 3, 3),
    ],
)
def test_calculator_operations(operator, left, right, expected):
    assert calculator(operator, left, right) == expected


def test_calculator_rejects_invalid_operations():
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        calculator("divide", 1, 0)
    with pytest.raises(ValueError, match="Unsupported operator"):
        calculator("power", 2, 3)
