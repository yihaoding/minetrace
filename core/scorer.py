"""Scorer — the pluggable scoring-engine interface.

A Scorer turns a fitted set of experts/evidence into a NodeScore tree for a
query sample. This is the single seam the backend re-architecture pivots on:
three interchangeable tracks all satisfy this protocol and all emit the same
NodeScore contract, so the narrative / agent layer is unaffected by which
track is active.

  Track A  ExpertTreeScorer    = core.expert_tree.CompositeNode   (AUC-weighted linear)
  Track B  EBMScorer           = glass-box additive model         (accuracy)
  Track C  BeliefFusionScorer  = Dempster-Shafer + BK revision    (uncertainty / novelty)

CompositeNode already satisfies this protocol structurally (fit / score_node),
so no wrapper is required for Track A; it is named here only to document the
seam.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from core.expert_tree import CompositeNode, NodeScore


@runtime_checkable
class Scorer(Protocol):
    """Any object that can be fitted on labelled data and score a sample."""

    def fit(self, positives: list, background: list) -> None: ...

    def score_node(self, sample) -> Optional[NodeScore]: ...


# Track A is the existing CompositeNode; expose an alias so call sites can name
# the abstraction without depending on the concrete class name.
ExpertTreeScorer = CompositeNode
