import pytest

from auditgym.calc import CalcError, evaluate


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("0.1 + 0.2", "0.3"),
        ("125000 * 1.0842", "135525"),
        ("(10 - 4) / 4", "1.5"),
        ("-3 + +2", "-1"),
        ("1 / 3", "0.3333333333333333333333333333333333"),
        ("100", "100"),
        ("2.50 * 4", "10"),
    ],
)
def test_arithmetic(expression, expected):
    assert evaluate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "open('/etc/passwd')",
        "x + 1",
        "2 ** 10",
        "10 % 3",
        "7 // 2",
        "'a' * 3",
        "[1, 2]",
        "(lambda: 1)()",
        "1 if 1 else 2",
        "1 / 0",
        "1e50 * 1e50",
        "1j",
        "1 +",
        "+" * 400,
        "().__class__",
    ],
)
def test_rejects_anything_but_arithmetic(expression):
    with pytest.raises(CalcError):
        evaluate(expression)


def test_rejects_non_string():
    with pytest.raises(CalcError):
        evaluate(42)
