"""DataSource interface and registry.

A DataSource is any provider of feature vectors keyed by sample ID.
Examples: GSWA site catalog, NGSA assay DB, MRI volumes, sensor archive.

Design rule: L1 Experts call get_features(id); they never read CSV directly.
Adding a new data source = new class implementing DataSource, registered here.
Core pipeline logic never changes.
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class DataSource(Protocol):
    """Uniform read interface for any data modality."""

    @property
    def name(self) -> str:
        """Unique registry key, e.g. 'site_catalog', 'ngsa_assay'."""
        ...

    @property
    def feature_names(self) -> list[str]:
        """All feature names this source can provide."""
        ...

    def get_features(self, sample_id: str) -> dict[str, float] | None:
        """Return feature dict for sample_id, or None if unavailable."""
        ...

    def is_available_for(self, sample_id: str) -> bool:
        """Fast check — avoid full get_features just to test availability."""
        ...


class DataSourceRegistry:
    """Central registry: sources register themselves by name."""

    _sources: dict[str, DataSource] = {}

    @classmethod
    def register(cls, source: DataSource) -> None:
        if source.name in cls._sources:
            logger.warning("Overwriting DataSource '%s'", source.name)
        cls._sources[source.name] = source
        logger.info("Registered DataSource '%s'", source.name)

    @classmethod
    def get(cls, name: str) -> DataSource:
        if name not in cls._sources:
            raise KeyError(f"DataSource '{name}' not registered. Available: {list(cls._sources)}")
        return cls._sources[name]

    @classmethod
    def list_available(cls) -> list[str]:
        return list(cls._sources.keys())

    @classmethod
    def clear(cls) -> None:
        """For testing only — wipe registry state."""
        cls._sources.clear()
