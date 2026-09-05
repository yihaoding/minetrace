"""P1–P5: Domain-agnostic abstract interfaces.

No geochem/medical/timeseries specifics here. x/y/km/Cu must NOT appear.
Concrete semantics (spatial coords vs voxel vs time index) live in domains/.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Sample(Protocol):
    """P1: A unit of data to be scored.

    Domain defines what 'identity' and 'features' mean:
    - geochem: site_code, element assay values
    - medical: patient_id + voxel, imaging intensities
    - timeseries: series_id + timestamp, sensor readings
    """

    @property
    def id(self) -> str: ...

    @property
    def features(self) -> dict[str, float]:
        """Named numeric features available for this sample."""
        ...


@runtime_checkable
class Instance(Protocol):
    """P2: A labeled positive example (known anomaly/deposit/lesion).

    confidence: [0,1] — how certain this label is.
    metadata: domain-specific tags (commodity, disease_type, fault_class …).
    """

    @property
    def sample(self) -> Sample: ...

    @property
    def confidence(self) -> float: ...

    @property
    def metadata(self) -> dict: ...


class NeighborhoodFn(Protocol):
    """P3: Returns the local neighborhood of a sample within a collection.

    Domain controls what 'neighbor' means:
    - geochem: samples within km radius
    - medical: voxels in spatial proximity
    - timeseries: samples in a sliding window
    """

    def __call__(
        self,
        sample: Sample,
        all_samples: list[Sample],
        **kwargs,
    ) -> list[Sample]: ...


class BackgroundFn(Protocol):
    """P4: Draws a background set for KB estimation.

    Returns samples that are NOT labeled positive instances, used to
    estimate the null distribution for each L1 Expert.
    """

    def __call__(
        self,
        all_samples: list[Sample],
        instances: list[Instance],
        **kwargs,
    ) -> list[Sample]: ...


class Fingerprint(Protocol):
    """P5: A fixed-length numeric embedding of one Instance.

    vector: the embedding used for similarity search (L2).
    components: named breakdown so callers can interpret dimensions.
    """

    @property
    def vector(self) -> np.ndarray: ...

    @property
    def components(self) -> dict[str, float]: ...
