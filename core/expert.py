"""Expert interface and registry.

Each Expert implements one hypothesis about what makes a sample anomalous.
Experts declare required_sources; if any source is missing from the registry
the Expert is silently skipped — callers never need to check manually.

Adding a new Expert = new file in domains/<domain>/experts/, register it.
Core pipeline code never changes.
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from core.primitives import Sample

logger = logging.getLogger(__name__)


@runtime_checkable
class Expert(Protocol):
    """L1 Expert: estimates P(anomaly | sample) from one or more DataSources."""

    @property
    def name(self) -> str: ...

    @property
    def required_sources(self) -> list[str]:
        """DataSource names this expert needs. Missing = auto-skip."""
        ...

    def fit_kb(self, background_samples: list[Sample]) -> None:
        """Estimate null distribution parameters from background samples."""
        ...

    def score(self, sample: Sample) -> float | None:
        """
        Return anomaly score in [0, 1] (higher = more anomalous).
        Return None if any required source is unavailable for this sample.
        """
        ...


class ExpertRegistry:
    """Collects all registered Experts; pipeline queries this at runtime."""

    _experts: dict[str, Expert] = {}

    @classmethod
    def register(cls, expert: Expert) -> None:
        cls._experts[expert.name] = expert
        logger.info("Registered Expert '%s' (needs: %s)", expert.name, expert.required_sources)

    @classmethod
    def get_active(cls, available_sources: list[str]) -> list[Expert]:
        """Return experts whose required_sources are all available."""
        active = []
        for exp in cls._experts.values():
            missing = [s for s in exp.required_sources if s not in available_sources]
            if missing:
                logger.debug("Expert '%s' skipped — missing sources: %s", exp.name, missing)
            else:
                active.append(exp)
        return active

    @classmethod
    def list_all(cls) -> list[str]:
        return list(cls._experts.keys())

    @classmethod
    def clear(cls) -> None:
        """For testing only."""
        cls._experts.clear()
