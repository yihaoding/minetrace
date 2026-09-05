"""Background sampling for geochem domain.

Implements core.BackgroundFn: returns samples that are NOT labeled instances,
used by L1 Experts to estimate the null (background) distribution.
"""
from __future__ import annotations

import random

from core.primitives import Instance, Sample


class ExcludeInstancesBackground:
    """Returns all samples that are not labeled positive instances.

    Optionally caps the background at `max_samples` (random subsample,
    seeded for reproducibility).
    """

    def __init__(self, max_samples: int | None = None, seed: int = 42) -> None:
        self.max_samples = max_samples
        self._rng = random.Random(seed)

    def __call__(
        self,
        all_samples: list[Sample],
        instances: list[Instance],
        **kwargs,
    ) -> list[Sample]:
        instance_ids = {inst.sample.id for inst in instances}
        background = [s for s in all_samples if s.id not in instance_ids]
        if self.max_samples is not None and len(background) > self.max_samples:
            background = self._rng.sample(background, self.max_samples)
        return background
