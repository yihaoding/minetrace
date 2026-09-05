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
    """Score returned by any node in the expert tree."""
    name:     str
    score:    float                               # [0, 1]
    confidence: float = 1.0                       # effective weight fraction at this query point
    children: dict[str, "NodeScore"] = field(default_factory=dict)
    weights:  dict[str, float]       = field(default_factory=dict)
    feature_z: dict[str, float]      = field(default_factory=dict)

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
    """

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

            s = ns.score
            if getattr(child, "_invert_score", False):
                s = 1.0 - s
            total_s += eff_w * s
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
