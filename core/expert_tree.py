"""Expert tree — AUC-weighted, locally-routed, traceable scoring.

NodeScore     : score tree (one node per expert) for full traceability.
CompositeNode : aggregates children with two-factor weighting:
                  global_weight  = (AUC_training − 0.5) × 2  [fitted once]
                  local_reliability = expert.local_reliability(sample)  [per query]
                  effective_weight  = global_weight × local_reliability
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


@dataclass
class NodeScore:
    """Score returned by any node in the expert tree.

    Sign convention (see CompositeNode): `score` is always the node's FINAL
    output — if the node is inverted, the inversion is already applied here and
    no ancestor applies it again. `feature_z` values are likewise already
    direction- and inversion-adjusted, so a leaf's raw pre-sigmoid score is
    recovered as sum(feature_w[k] * feature_z[k]) with no further sign flips.
    """
    name:     str
    score:    float                               # [0, 1]
    confidence: float = 1.0                       # effective weight fraction at this query point
    children: dict[str, "NodeScore"] = field(default_factory=dict)
    weights:  dict[str, float]       = field(default_factory=dict)
    feature_z: dict[str, float]      = field(default_factory=dict)
    # Per-feature effective weight, keyed exactly like feature_z (multi-source
    # leaves use "src:feat"), normalized to sum to 1 within the leaf. `weights`
    # cannot serve this role: on multi-source leaves it is keyed by SOURCE name.
    feature_w: dict[str, float]      = field(default_factory=dict)
    # Dempster-Shafer extension (populated only by belief-fusion scorers).
    # Left as None by the default linear/AUC tree so existing consumers are
    # unaffected; when present, (bel, pl) give the [Bel, Pl] uncertainty
    # interval for the "prospective" hypothesis at this node.
    bel: float | None = None
    pl:  float | None = None

    def flat_scores(self) -> dict[str, float]:
        out = {self.name: self.score}
        for child in self.children.values():
            out.update(child.flat_scores())
        return out

    def active_experts(self) -> dict[str, tuple[float, float]]:
        """Return {expert_name: (score, confidence)} for all leaf children."""
        out: dict[str, tuple[float, float]] = {}
        for name, child in self.children.items():
            if not child.children:
                out[name] = (child.score, child.confidence)
            else:
                out.update(child.active_experts())
        return out

    def all_feature_z(self) -> dict[str, tuple[float, str]]:
        out: dict[str, tuple[float, str]] = {}
        for feat, z in self.feature_z.items():
            out[feat] = (z, self.name)
        for child in self.children.values():
            out.update(child.all_feature_z())
        return out

    def all_feature_signals(self) -> dict[str, tuple[float, float, str]]:
        """Recursively collect {feature_key: (z, weight, owning_expert_name)}.

        Pulls z and weight from the SAME node, so the pairing is always correct
        — unlike looking up a leaf's feature key in an ancestor's `weights`.
        """
        out: dict[str, tuple[float, float, str]] = {}
        for feat, z in self.feature_z.items():
            w = self.feature_w.get(feat, self.weights.get(feat, 0.0))
            out[feat] = (z, float(w), self.name)
        for child in self.children.values():
            out.update(child.all_feature_signals())
        return out


class CompositeNode:
    """Aggregates child experts with routing-aware weighting.

    Fit
    ---
    Calls fit() on every child, then computes each child's training AUC
    → global_weight = max(0, (AUC − 0.5) × 2).

    Score
    -----
    effective_weight = global_weight × child.local_reliability(sample)

    local_reliability defaults to 1.0 if the expert does not implement it.
    This means experts can silently down-weight themselves when they have
    poor local data coverage — no changes needed to this class.

    Inversion
    ---------
    A node with AUC < 0.5 is marked `_invert_score = True` by its PARENT at fit
    time, but the inversion is applied by the node ITSELF at score time — every
    node owns exactly one flip of its own output. A parent must never invert a
    child's score, or the child's internal flip and the parent's flip cancel and
    the anti-correlated signal enters the aggregate with a positive weight.
    """

    _invert_score: bool = False

    def __init__(self, name: str, children: list) -> None:
        self.name = name
        self.children: list = children
        self._child_weights: dict[str, float] = {}
        self._fitted = False

    # ── fitting ───────────────────────────────────────────────────────────────

    def fit(self, positives: list, background: list) -> None:
        for child in self.children:
            child.fit(positives, background)

        all_samples = positives + background
        y = [1] * len(positives) + [0] * len(background)

        for child in self.children:
            scores, valid_y = [], []
            for s, label in zip(all_samples, y):
                ns = child.score_node(s)
                if ns is not None:
                    scores.append(ns.score)
                    valid_y.append(label)

            if len(set(valid_y)) < 2 or len(scores) < 10:
                logger.warning("CompositeNode %s: %s too few samples → weight=0",
                               self.name, child.name)
                self._child_weights[child.name] = 0.0
                continue

            try:
                auc = float(roc_auc_score(valid_y, scores))
            except Exception:
                auc = 0.5

            if auc < 0.5:
                auc = 1.0 - auc
                child._invert_score = True
            else:
                child._invert_score = getattr(child, "_invert_score", False)

            w = max(0.0, (auc - 0.5) * 2.0)
            self._child_weights[child.name] = w
            logger.info("  %s / %s  AUC=%.3f  weight=%.3f",
                        self.name, child.name, auc, w)

        self._fitted = True

    # ── scoring ───────────────────────────────────────────────────────────────

    def score_node(self, sample) -> NodeScore | None:
        child_scores: dict[str, NodeScore] = {}
        for child in self.children:
            ns = child.score_node(sample)
            if ns is not None:
                child_scores[child.name] = ns

        if not child_scores:
            return None

        total_w = total_s = 0.0
        eff_weights: dict[str, float] = {}

        for child in self.children:
            if child.name not in child_scores:
                continue
            ns = child_scores[child.name]
            global_w  = self._child_weights.get(child.name, 0.0)
            local_r   = getattr(child, "local_reliability", lambda _: 1.0)(sample)
            eff_w     = global_w * float(local_r)
            eff_weights[child.name] = eff_w

            # ns.score is the child's FINAL score — if the child is inverted it
            # already flipped it. Do not flip again here.
            total_s += eff_w * ns.score
            total_w += eff_w

        if total_w < 1e-9:
            g = float(np.mean([ns.score for ns in child_scores.values()]))
            confidence = 0.0
        else:
            g = total_s / total_w
            # confidence = fraction of max-possible weight actually used
            max_possible = sum(self._child_weights.get(c.name, 0.0)
                               for c in self.children)
            confidence = float(total_w / max_possible) if max_possible > 1e-9 else 1.0

        # this node owns its own inversion (see class docstring)
        if self._invert_score:
            g = 1.0 - g

        # annotate each child NodeScore with its effective weight fraction
        for cname, ns in child_scores.items():
            ns.confidence = eff_weights.get(cname, 0.0)

        return NodeScore(
            name=self.name,
            score=float(np.clip(g, 0.0, 1.0)),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            children=child_scores,
            weights=eff_weights,
        )
