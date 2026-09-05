"""DataCatalog — metadata registry of available data sources.

SourceSpec describes what a source is (type, modality, subtype, confidence, path).
DataCatalog holds a collection of SourceSpecs.
AnalysisAgent reads this to decide what to load and activate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import nan
from typing import List


@dataclass
class SourceSpec:
    """Metadata description of one data source.

    Attributes
    ----------
    name         : unique registry key used by DataSourceRegistry
    source_type  : "assay_spatial" | "raster" | "vector" | "timeseries" | "image"
    modality     : "geochemistry" | "geophysics" | "geology" | domain string
    subtype      : finer grain, e.g. "sediment" | "drillhole" | "soil" | "magnetic"
    confidence   : prior reliability weight [0, 1]
    resolution_km: nominal spatial resolution (NaN if not applicable)
    path         : file-system path to the underlying data
    loader_kwargs: extra keyword args forwarded to the DataSource constructor
    description  : human-readable note
    """

    name: str
    source_type: str
    modality: str
    subtype: str = ""
    confidence: float = 1.0
    resolution_km: float = nan
    path: str = ""
    loader_kwargs: dict = field(default_factory=dict)
    description: str = ""


class DataCatalog:
    """Registry of SourceSpec objects.

    Usage::

        catalog = DataCatalog()
        catalog.register(SourceSpec(name="sediment", source_type="assay_spatial",
                                    modality="geochemistry", subtype="sediment",
                                    path="/path/to/sediment.csv"))
        spec = catalog.get("sediment")
    """

    def __init__(self) -> None:
        self._specs: dict[str, SourceSpec] = {}

    def register(self, spec: SourceSpec) -> None:
        """Add or overwrite a SourceSpec."""
        self._specs[spec.name] = spec

    def get(self, name: str) -> SourceSpec:
        """Return the SourceSpec with the given name.

        Raises KeyError if not registered.
        """
        if name not in self._specs:
            raise KeyError(
                f"SourceSpec '{name}' not in catalog. "
                f"Available: {list(self._specs)}"
            )
        return self._specs[name]

    def list_by_modality(self, modality: str) -> List[SourceSpec]:
        """Return all specs with the given modality."""
        return [s for s in self._specs.values() if s.modality == modality]

    def list_by_source_type(self, source_type: str) -> List[SourceSpec]:
        """Return all specs with the given source_type."""
        return [s for s in self._specs.values() if s.source_type == source_type]

    def all(self) -> List[SourceSpec]:
        """Return all registered specs."""
        return list(self._specs.values())

    def summary(self) -> str:
        """One-line per spec summary string."""
        lines = [f"DataCatalog ({len(self._specs)} sources):"]
        for spec in self._specs.values():
            lines.append(
                f"  {spec.name:<25s}  type={spec.source_type:<15s}"
                f"  modality={spec.modality:<15s}  subtype={spec.subtype}"
            )
        return "\n".join(lines)
