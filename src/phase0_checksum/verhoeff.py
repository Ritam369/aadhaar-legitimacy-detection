"""Aadhaar number validation: format rules + Verhoeff checksum.

A pass only means the number is well-formed, NOT that it was ever issued.
"""
import re

# Multiplication table (dihedral group D5)
_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
# Permutation table
_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def verhoeff_checksum(digits: str) -> int:
    """Return 0 if `digits` (including its check digit) is valid."""
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c


def validate_aadhaar_number(raw: str):
    """Return (is_valid, reason). Spaces/hyphens are allowed as separators."""
    s = re.sub(r"[\s-]", "", raw or "")
    if not s.isdigit():
        return False, "Contains non-digit characters"
    if len(s) != 12:
        return False, f"Must be 12 digits, got {len(s)}"
    if s[0] in "01":
        return False, "First digit cannot be 0 or 1"
    if verhoeff_checksum(s) != 0:
        return False, "Verhoeff checksum failed"
    return True, "Well-formed (checksum OK); does not prove the number was issued"
