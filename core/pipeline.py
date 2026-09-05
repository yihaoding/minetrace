"""L1 pipeline — domain-agnostic Expert orchestration.

Responsibilities:
  1. Query ExpertRegistry for all registered Experts
  2. Filter to those whose required_sources are all available
  3. fit_kb each active Expert on the background samples
  4. Compute G(x) = mean of z-scored individual Expert scores

No geochem/medical/timeseries specifics here.
x, y, km, Cu must NOT appear in this file.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from core.data_source import DataSourceRegistry
from core.expert import Expert, ExpertRegistry
from core.primitives import Instance, Sample

logger = logging.getLogger(__name__)


@dataclass
class ScoredSample:
    """Output record for one sample from the L1 pipeline."""

    sample: Sample
    g_score: float                          # combined anomaly score [0, 1]
    expert_scores: dict[str, float] = field(default_factory=dict)  # per-expert raw scores
    expert_z_scores: dict[str, float] = field(default_factory=dict)


class L1Pipeline:
    """Domain-agnostic L1 anomaly scoring pipeline.

    Usage:
        pipeline = L1Pipeline()
        pipeline.fit(all_samples, instances)
        results = pipeline.score_all(all_samples)
    """

    def __init__(self) -> None:
        self._active_experts: list[Expert] = []
        self._expert_bg_mean: dict[str, float] = {}
        self._expert_bg_std: dict[str, float] = {}
        self._fitted = False

    def fit(
        self,
        all_samples: list[Sample],
        instances: list[Instance],
        background_fn=None,
    ) -> None:
        """Fit all active Experts on background samples.

        background_fn: callable(all_samples, instances) → list[Sample].
        Defaults to ExcludeInstancesBackground if None.
        """
        available = DataSourceRegistry.list_available()
        self._active_experts = ExpertRegistry.get_active(available)
        logger.info(
            "L1Pipeline: %d/%d experts active (available sources: %s)",
            len(self._active_experts),
            len(ExpertRegistry.list_all()),
            available,
        )

        if background_fn is None:
            from domains.geochem.background import ExcludeInstancesBackground
            background_fn = ExcludeInstancesBackground()

        background = background_fn(all_samples, instances)
        logger.info("Background set: %d samples", len(background))

        for expert in self._active_experts:
            expert.fit_kb(background)

        # Calibrate z-score normalization using background scores
        for expert in self._active_experts:
            bg_scores = []
            for s in background:
                sc = expert.score(s)
                if sc is not None:
                    bg_scores.append(sc)
            arr = np.array(bg_scores)
            self._expert_bg_mean[expert.name] = float(arr.mean()) if len(arr) > 0 else 0.5
            self._expert_bg_std[expert.name] = float(arr.std()) if len(arr) > 1 else 1.0
            if self._expert_bg_std[expert.name] < 1e-9:
                self._expert_bg_std[expert.name] = 1.0

        self._fitted = True

    def score(self, sample: Sample) -> ScoredSample:
        """Score a single sample."""
        expert_scores: dict[str, float] = {}
        z_scores: dict[str, float] = {}

        for expert in self._active_experts:
            raw = expert.score(sample)
            if raw is None:
                continue
            expert_scores[expert.name] = raw
            z = (raw - self._expert_bg_mean[expert.name]) / self._expert_bg_std[expert.name]
            z_scores[expert.name] = z

        if z_scores:
            mean_z = float(np.mean(list(z_scores.values())))
            # Re-sigmoid to keep G(x) in [0, 1]
            g = float(1.0 / (1.0 + np.exp(-mean_z)))
        else:
            g = 0.5  # no active experts → neutral

        return ScoredSample(
            sample=sample,
            g_score=g,
            expert_scores=expert_scores,
            expert_z_scores=z_scores,
        )

    def score_all(self, samples: list[Sample]) -> list[ScoredSample]:
        """Score every sample in the list."""
        return [self.score(s) for s in samples]

    @property
    def active_expert_names(self) -> list[str]:
        return [e.name for e in self._active_experts]
