"""Regression tests for IP-01 / IP-02 / IP-03.

IP-01  double inversion  — a leaf with AUC < 0.5 flipped its own raw score AND
                           its parent CompositeNode flipped again (1-s), so the
                           two cancelled: the anti-correlated signal entered the
                           aggregate un-inverted while the narrative displayed
                           the inverted score.
IP-02  weight lookup     — multi-source leaves key feature_z by "src:feat" but
                           key `weights` by source name, so the narrative's
                           weights.get(feature) returned 0 and every displayed
                           contribution (weight × z) collapsed to 0.
IP-03  zip misalignment  — `zip(self._states, sub_weights)` paired source names
                           with the wrong weights whenever a source was skipped.

Invariant now enforced everywhere: every node applies its own `_invert_score`
inside score_node() and no ancestor re-applies it; feature_z is stored already
inversion-adjusted; feature_w is keyed identically to feature_z.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.expert_tree import CompositeNode, NodeScore


class _Sample:
    def __init__(self, sid: str, label: int) -> None:
        self.id = sid
        self.label = label


class _FakeLeaf:
    """Leaf whose score is a sigmoid of a weighted z-sum, honouring _invert_score.

    Mirrors the real leaves' contract: it flips its own raw score, flips its
    feature_z with it, and publishes feature_w keyed exactly like feature_z.
    """

    _invert_score = False

    def __init__(self, name: str, z_by_id: dict[str, dict[str, float]]) -> None:
        self.name = name
        self._z = z_by_id

    def fit(self, positives, background):  # noqa: D102
        pass

    def score_node(self, sample):
        z = self._z.get(sample.id)
        if z is None:
            return None
        w = {f: 1.0 / len(z) for f in z}
        if self._invert_score:
            z = {f: -v for f, v in z.items()}
        raw = sum(w[f] * z[f] for f in z)
        return NodeScore(
            name=self.name,
            score=float(1.0 / (1.0 + np.exp(-raw))),
            weights=dict(w),
            feature_z=z,
            feature_w=w,
        )

    def score(self, sample):
        ns = self.score_node(sample)
        return ns.score if ns is not None else None


def _anti_correlated_data(n: int = 40):
    """Positives get LOW feature values → the leaf's raw AUC is < 0.5."""
    pos = [_Sample(f"p{i}", 1) for i in range(n)]
    bg = [_Sample(f"b{i}", 0) for i in range(n)]
    z = {}
    for i, s in enumerate(pos):
        z[s.id] = {"f1": -2.0 - 0.01 * i, "f2": -1.5 - 0.01 * i}
    for i, s in enumerate(bg):
        z[s.id] = {"f1": 2.0 + 0.01 * i, "f2": 1.5 + 0.01 * i}
    return pos, bg, z


def test_inverted_leaf_is_flipped_exactly_once():
    """IP-01: an anti-correlated leaf must SEPARATE positives after fitting.

    Pre-fix the leaf's flip and the parent's flip cancelled, so the aggregate
    ranked positives BELOW background (root AUC < 0.5) despite a positive weight.
    """
    pos, bg, z = _anti_correlated_data()
    leaf = _FakeLeaf("anti", z)
    root = CompositeNode("root", [leaf])
    root.fit(pos, bg)

    assert leaf._invert_score is True, "parent should have marked the leaf inverted"

    pos_g = [root.score_node(s).score for s in pos]
    bg_g = [root.score_node(s).score for s in bg]
    assert min(pos_g) > max(bg_g), (
        "inverted leaf must raise the root score on positives; "
        "if the flip is applied twice it cancels and this reverses"
    )


def test_leaf_score_equals_what_parent_aggregates():
    """IP-01: the score the narrative displays IS the score the tree aggregates."""
    pos, bg, z = _anti_correlated_data()
    leaf = _FakeLeaf("anti", z)
    root = CompositeNode("root", [leaf])
    root.fit(pos, bg)

    for s in pos + bg:
        ns = root.score_node(s)
        # single child ⇒ the weighted mean collapses to that child's own score
        assert ns.score == pytest.approx(ns.children["anti"].score, abs=1e-12)


def test_feature_z_sign_tracks_the_inverted_score():
    """IP-01: feature_z must be flipped with the score it explains.

    Otherwise the narrative shows "z = +3.1, therefore anomalous" attached to a
    score that was computed from -3.1.
    """
    pos, bg, z = _anti_correlated_data()
    leaf = _FakeLeaf("anti", z)
    root = CompositeNode("root", [leaf])
    root.fit(pos, bg)

    ns = root.score_node(pos[0]).children["anti"]
    assert ns.score > 0.5, "positive should score high after inversion"
    # raw values for positives were negative; after inversion they must read positive
    assert all(v > 0 for v in ns.feature_z.values()), (
        "feature_z still carries the pre-inversion sign — it contradicts the score"
    )


def test_feature_w_keys_match_feature_z_keys():
    """IP-02: every feature_z key must resolve to a weight, or the contribution is 0."""
    pos, bg, z = _anti_correlated_data()
    leaf = _FakeLeaf("anti", z)
    root = CompositeNode("root", [leaf])
    root.fit(pos, bg)

    signals = root.score_node(pos[0]).all_feature_signals()
    assert signals, "expected feature signals from the leaf"
    for key, (zv, w, _owner) in signals.items():
        assert w > 0.0, f"feature {key!r} resolved to weight 0 — narrative would drop it"


def test_additivity_leaf_raw_reconstructs_from_feature_w_dot_z():
    """The leaf's pre-sigmoid raw must equal Σ feature_w · feature_z, no sign flips."""
    pos, bg, z = _anti_correlated_data()
    leaf = _FakeLeaf("anti", z)
    root = CompositeNode("root", [leaf])
    root.fit(pos, bg)

    for s in pos[:5] + bg[:5]:
        ns = root.score_node(s).children["anti"]
        raw = sum(ns.feature_w[f] * ns.feature_z[f] for f in ns.feature_z)
        assert float(1.0 / (1.0 + np.exp(-raw))) == pytest.approx(ns.score, abs=1e-12)
