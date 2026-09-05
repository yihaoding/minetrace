"""AnalysisAgent — domain-agnostic agentic analysis protocol.

An AnalysisAgent is any object that can:
  1. Accept a DataCatalog + target metadata and plan / load its data sources.
  2. Score individual or batches of Sample objects using a fitted expert tree.

AnalysisPlan records which sources were activated and why — useful for
audit trails and debugging.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, runtime_checkable

from core.expert_tree import NodeScore
from core.primitives import Sample


@runtime_checkable
class AnalysisAgent(Protocol):
    """Domain-agnostic agentic analysis interface.

    Concrete implementations (e.g. GeochemAgent) read a DataCatalog,
    decide which DataSources and Experts to activate, load and fit
    everything, then expose score() / score_batch() for inference.
    """

    def setup(self, catalog, target_meta: dict) -> None:
        """Load data sources, build and fit the expert tree.

        Parameters
        ----------
        catalog     : DataCatalog describing available data
        target_meta : domain-specific configuration dict
        """
        ...

    def score(self, sample: Sample) -> Optional[NodeScore]:
        """Score a single pre-registered sample."""
        ...

    def score_batch(self, samples: List[Sample]) -> list:
        """Register and score a list of samples, returning NodeScore or None each."""
        ...

    def active_sources(self) -> List[str]:
        """Names of DataSources that are currently loaded and active."""
        ...

    def active_experts(self) -> List[str]:
        """Names of Experts in the current expert tree."""
        ...


@dataclass
class AnalysisPlan:
    """Record of which sources the agent decided to activate and why.

    Produced by GeochemAgent.plan() before any loading occurs.
    Useful for dry-runs, logging, and reproducibility.

    Attributes
    ----------
    active_source_names   : sources that will be loaded
    expert_tree_description: short label per expert node that will be built
    rationale             : source_name → reason string (activated or skipped)
    """

    active_source_names: List[str] = field(default_factory=list)
    expert_tree_description: List[str] = field(default_factory=list)
    rationale: dict = field(default_factory=dict)  # str → str

    def summary(self) -> str:
        lines = [
            f"AnalysisPlan: {len(self.active_source_names)} active sources, "
            f"{len(self.expert_tree_description)} expert nodes",
            "",
            "Active sources:",
        ]
        for name in self.active_source_names:
            lines.append(f"  + {name}")
        lines.append("")
        lines.append("Rationale:")
        for src, reason in self.rationale.items():
            prefix = "  [ACTIVE]  " if src in self.active_source_names else "  [SKIP]   "
            lines.append(f"{prefix}{src}: {reason}")
        lines.append("")
        lines.append("Expert tree:")
        for desc in self.expert_tree_description:
            lines.append(f"  {desc}")
        return "\n".join(lines)
