"""BeliefFunction: domain-agnostic Dempster-Shafer structure.

A BeliefFunction lives on a frame of discernment Ω (a finite set of
mutually exclusive hypotheses). Mass is assigned to subsets (focal elements).

Design rule: no geochem/medical terms here. frame elements are arbitrary
strings; callers define what they mean.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BeliefFunction:
    """Dempster-Shafer belief function on a finite frame Ω.

    frame: ordered list of singletons in Ω.
    mass:  mapping focal_element -> mass value. Keys are frozensets of strings
           drawn from frame. Must sum to 1.0 (validated on construction).
    """

    frame: list[str]
    mass: dict[frozenset[str], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        total = sum(self.mass.values())
        if self.mass and abs(total - 1.0) > 1e-6:
            raise ValueError(f"Mass values must sum to 1.0, got {total:.6f}")
        frame_set = set(self.frame)
        for focal in self.mass:
            unknown = focal - frame_set
            if unknown:
                raise ValueError(f"Focal element {focal} contains elements not in frame: {unknown}")

    # ------------------------------------------------------------------
    # Derived measures
    # ------------------------------------------------------------------
    def bel(self, event: frozenset[str]) -> float:
        """Belief: sum of mass over all focal elements ⊆ event."""
        return sum(m for A, m in self.mass.items() if A and A.issubset(event))

    def pl(self, event: frozenset[str]) -> float:
        """Plausibility: sum of mass over focal elements that intersect event."""
        return sum(m for A, m in self.mass.items() if A and A & event)

    def doubt(self, event: frozenset[str]) -> float:
        """Doubt: belief in the complement of event."""
        complement = frozenset(self.frame) - event
        return self.bel(complement)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def focal_elements(self) -> list[frozenset[str]]:
        return [A for A, m in self.mass.items() if m > 0]

    def vacuous(self) -> bool:
        """True if all mass is on the full frame (no information)."""
        full = frozenset(self.frame)
        return abs(self.mass.get(full, 0.0) - 1.0) < 1e-9

    def to_pignistic(self) -> dict[str, float]:
        """BetP: pignistic probability transform for decision making."""
        bet = {h: 0.0 for h in self.frame}
        for A, m in self.mass.items():
            if not A:
                continue
            share = m / len(A)
            for h in A:
                bet[h] += share
        return bet

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{{{','.join(sorted(A))}}}: {m:.4f}" for A, m in self.mass.items() if m > 0
        )
        return f"BeliefFunction(frame={self.frame}, mass={{{parts}}})"


# ------------------------------------------------------------------
# Convenience constructors
# ------------------------------------------------------------------

def categorical_mass(frame: list[str], singleton: str, certainty: float = 1.0) -> BeliefFunction:
    """All mass on one singleton (and remainder on full frame)."""
    if singleton not in frame:
        raise ValueError(f"'{singleton}' not in frame {frame}")
    remainder = 1.0 - certainty
    mass: dict[frozenset[str], float] = {frozenset([singleton]): certainty}
    if remainder > 1e-9:
        mass[frozenset(frame)] = remainder
    return BeliefFunction(frame=frame, mass=mass)


def vacuous_mass(frame: list[str]) -> BeliefFunction:
    """Complete ignorance: all mass on full frame."""
    return BeliefFunction(frame=frame, mass={frozenset(frame): 1.0})
