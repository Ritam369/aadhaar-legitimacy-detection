"""Shared output format: every phase returns a PhaseResult."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseResult:
    score: float   # 0.0 = looks fake, 1.0 = looks genuine
    reason: str

    def __post_init__(self):
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")

    def as_tuple(self):
        return (self.score, self.reason)
