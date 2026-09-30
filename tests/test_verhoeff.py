import pytest
from src.phase0_checksum.verhoeff import validate_aadhaar_number


@pytest.mark.parametrize("num,expected", [
    ("234567890124", True),     # constructed to pass
    ("2345 6789 0124", True),   # spaces allowed
    ("234567890125", False),    # last digit changed
    ("123456789012", False),    # starts with 1
    ("023456789012", False),    # starts with 0
    ("2345 6789 012", False),   # only 11 digits
    ("23456789012A", False),    # non-digit
    ("", False),
])
def test_numbers(num, expected):
    assert validate_aadhaar_number(num)[0] is expected


def test_single_digit_error_always_caught():
    base = "234567890124"
    for i in range(12):
        for d in "0123456789":
            if d != base[i]:
                bad = base[:i] + d + base[i + 1:]
                if bad[0] in "01":
                    continue
                assert validate_aadhaar_number(bad)[0] is False
