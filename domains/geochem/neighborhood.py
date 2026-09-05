"""Spatial neighborhood function for geochem domain.

Implements core.NeighborhoodFn: returns samples within a km radius.
Uses approximate degree conversion (accurate enough for WA extent).
"""
from __future__ import annotations

import math

from domains.geochem.samples import GeochemSample

_KM_PER_DEG_LAT = 111.0


def _haversine_approx(x1: float, y1: float, x2: float, y2: float) -> float:
    """Great-circle distance in km (fast approximation for small distances)."""
    dlat = math.radians(y2 - y1)
    dlon = math.radians(x2 - x1)
    lat_mid = math.radians((y1 + y2) / 2)
    a = dlat**2 + (math.cos(lat_mid) * dlon) ** 2
    return math.sqrt(a) * _KM_PER_DEG_LAT


class SpatialNeighborhood:
    """Returns all samples within `radius_km` of the query sample.

    Implements core.NeighborhoodFn via __call__.
    Only works with GeochemSample (has .x, .y attributes).
    """

    def __init__(self, radius_km: float = 20.0) -> None:
        self.radius_km = radius_km

    def __call__(
        self,
        sample: GeochemSample,
        all_samples: list[GeochemSample],
        **kwargs,
    ) -> list[GeochemSample]:
        neighbors = []
        for s in all_samples:
            if s.id == sample.id:
                continue
            if _haversine_approx(sample.x, sample.y, s.x, s.y) <= self.radius_km:
                neighbors.append(s)
        return neighbors
