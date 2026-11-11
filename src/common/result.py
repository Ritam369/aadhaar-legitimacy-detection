"""Shared output format: every phase returns a PhaseResult."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PhaseResult:
    score: float   # 0.0 = looks fake, 1.0 = looks genuine
    reason: str
    # Sub-scores / flags for Phase 7 and for evaluation (not used in comparisons).
    details: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")

    def as_tuple(self):
        return (self.score, self.reason)

    @classmethod
    def neutral(cls, reason, details=None):
        """Could not check (masked number, card not located, API down...).
        This is NOT evidence of forgery, so it scores 0.5."""
        return cls(0.5, reason, details or {})