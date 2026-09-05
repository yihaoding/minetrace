"""IP-02 / IP-03 regression tests against the real multi-source geochem experts.

IP-03 is only observable when a source is SKIPPED at score time (source_weight
below threshold, or no coverage at the query point): the old
`zip(self._states, sub_weights)` then paired the surviving weights with the
wrong source names, silently attributing one source's weight to another.
"""
from __future__ import annotations

import pytest

from core.data_source import DataSourceRegistry
from domains.geochem.experts.geochem_experts import TargetEnrichmentExpert


class _FakeSource:
    """Minimal DataSource: serves a fixed feature dict per sample id."""

    def __init__(self, name: str, rows: dict[str, dict[str, float]]) -> None:
        self._name = name
        self._rows = rows

    @property
    def name(self) -> str:
        return self._name

    @property
    def feature_names(self) -> list[str]:
        return sorted({f for r in self._rows.values() for f in r})

    def get_features(self, sample_id: str):
        return self._rows.get(sample_id)

    def is_available_for(self, sample_id: str) -> bool:
        return sample_id in self._rows


class _Sample:
    def __init__(self, sid: str) -> None:
        self.id = sid


_STATS = ["contrast_5_50", "contrast_10_50",
          "5km_frac_above", "10km_frac_above", "5km_p90", "10km_p90"]


def _rows(ids, value):
    return {i: {f"Cu_{s}": value for s in _STATS} for i in ids}


@pytest.fixture
def two_sources():
    """'good' separates classes; 'noise' does not (→ source_weight 0, skipped)."""
    pos_ids = [f"p{i}" for i in range(30)]
    bg_ids = [f"b{i}" for i in range(30)]

    good_rows = {}
    good_rows.update({i: {f"Cu_{s}": 5.0 + 0.01 * n for s in _STATS}
                      for n, i in enumerate(pos_ids)})
    good_rows.update({i: {f"Cu_{s}": 0.0 + 0.01 * n for s in _STATS}
                      for n, i in enumerate(bg_ids)})

    # identical distributions for both classes → AUC 0.5 → weight 0 → skipped
    noise_rows = _rows(pos_ids + bg_ids, 1.0)

    good = _FakeSource("good", good_rows)
    noise = _FakeSource("noise", noise_rows)
    DataSourceRegistry.register(good)
    DataSourceRegistry.register(noise)
    yield pos_ids, bg_ids
    DataSourceRegistry._sources.pop("good", None)
    DataSourceRegistry._sources.pop("noise", None)


def test_weights_are_keyed_by_the_source_that_actually_scored(two_sources):
    """IP-03: a skipped source must not shift the weight dict onto the wrong key."""
    pos_ids, bg_ids = two_sources
    # 'noise' is listed FIRST, so the old zip() would pair 'noise' with the
    # weight that actually belongs to 'good'.
    expert = TargetEnrichmentExpert("Cu", ["noise", "good"])
    expert.fit([_Sample(i) for i in pos_ids], [_Sample(i) for i in bg_ids])

    ns = expert.score_node(_Sample(pos_ids[0]))
    assert ns is not None

    assert "noise" not in ns.weights, (
        "a source with weight 0 is skipped at score time and must not appear "
        "in the reported weights"
    )
    assert set(ns.weights) == {"good"}
    assert ns.weights["good"] == pytest.approx(expert._states["good"].source_weight)


def test_feature_z_and_feature_w_share_keys(two_sources):
    """IP-02: multi-source keys are "src:feat" — feature_w must use the same keys."""
    pos_ids, bg_ids = two_sources
    expert = TargetEnrichmentExpert("Cu", ["noise", "good"])
    expert.fit([_Sample(i) for i in pos_ids], [_Sample(i) for i in bg_ids])

    ns = expert.score_node(_Sample(pos_ids[0]))
    assert ns.feature_z, "expected per-feature z-scores"
    assert set(ns.feature_z) == set(ns.feature_w), (
        "feature_w must be keyed exactly like feature_z, or every displayed "
        "contribution (weight × z) collapses to 0"
    )
    assert all(k.startswith("good:") for k in ns.feature_z)
    assert sum(ns.feature_w.values()) == pytest.approx(1.0)

    # and the narrative's lookup path must now resolve a non-zero weight
    for key, (z, w, _owner) in ns.all_feature_signals().items():
        assert w > 0.0, f"{key!r} resolved to weight 0"
