"""GeochemSample and GeochemInstance — geochem domain implementations of core.Sample / core.Instance."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GeochemSample:
    """A single geochemical site — implements core.Sample.

    x, y are geographic coordinates (longitude, latitude).
    raw_features is populated by DataSources via the pipeline.
    """

    site_code: str
    x: float
    y: float
    site_commo: list[str] = field(default_factory=list)
    site_stage: str = ""
    site_type: str = ""
    raw_features: dict[str, float] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.site_code

    @property
    def features(self) -> dict[str, float]:
        return self.raw_features


@dataclass
class GeochemInstance:
    """A labeled high-confidence Cu positive site — implements core.Instance."""

    sample: GeochemSample
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


# DP-02 resolution: high-confidence Cu positive rule
# Cu_SITES contains "Mine" OR "Deposit" (752 sites total).
# "Prospect + Operating Stage" combination does not exist in the data
# (all Prospects have SITE_STAGE=Undeveloped), so that branch is not used.
STAGE_SCORE: dict[str, int] = {
    "Operating": 5,
    "Under Development": 4,
    "Proposed": 3,
    "Care and Maintenance": 2,
    "Shut": 1,
    "Undeveloped": 0,
}


def is_high_confidence_cu(cu_sites_value: str) -> bool:
    """Return True if a site is a high-confidence Cu positive (Mine or Deposit)."""
    if not isinstance(cu_sites_value, str):
        return False
    return "Mine" in cu_sites_value or "Deposit" in cu_sites_value
