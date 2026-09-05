"""Persist a fitted ProspectivityModel + its data sources to one pickle file.

Save side (compute node, once)::

    from domains.geochem.persistence import save_fitted
    save_fitted(model, "models/Cu_10expert_global.pkl")

Load side (backend service, at startup)::

    from domains.geochem.persistence import load_fitted
    fm = load_fitted("models/Cu_10expert_global.pkl")
    res = fm.score_point(x=120.5, y=-28.3)   # any new point, no re-fit
    res["global"]   # DS belief fusion score (reliability = global AUC weight)
    res["tree"]     # linear AUC-weighted tree score
    res["node"]     # full NodeScore tree (per-expert breakdown, for narrative)

New points are computed on the fly: each data source keeps its fitted state
(assay KD-trees, background stats, expert weights) in the pickle, and rasters
are re-opened from their original paths on load — so scoring a point the model
has never seen only costs the feature extraction for that point (~ms).
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

from core.data_source import DataSourceRegistry
from domains.geochem.belief_fusion import BeliefFusionScorer
from domains.geochem.samples import GeochemSample

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1


def save_fitted(model, path: str | Path) -> Path:
    """Pickle a fitted ProspectivityModel together with its active sources."""
    if not getattr(model, "_fitted", False):
        raise RuntimeError("Model is not fitted — call setup() first.")
    sources = {}
    for name in model.active_sources():
        sources[name] = DataSourceRegistry.get(name)
    bundle = {
        "format_version": FORMAT_VERSION,
        "target": model._config.target,
        "expert_names": model.active_experts(),
        "model": model,
        "sources": sources,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved fitted %s model (%d experts) -> %s",
                bundle["target"], len(bundle["expert_names"]), path)
    return path


class FittedModel:
    """Loaded bundle: fitted tree + sources + global-mode belief fusion scorer."""

    def __init__(self, bundle: dict) -> None:
        self.target: str = bundle["target"]
        self.expert_names: list[str] = bundle["expert_names"]
        self.model = bundle["model"]
        self.sources = bundle["sources"]
        for src in self.sources.values():
            DataSourceRegistry.register(src)
        # mode="global" needs no calibration — reliability comes straight from
        # the fitted tree's AUC child-weights.
        self.scorer = BeliefFusionScorer(self.model._tree, mode="global")

    def _register(self, samples: list) -> None:
        for src in self.sources.values():
            if hasattr(src, "register_samples"):
                src.register_samples(samples)

    def score_sample(self, sample: GeochemSample) -> dict:
        self._register([sample])
        ns_global = self.scorer.score_node(sample)
        ns_tree = self.model._tree.score_node(sample)
        return {
            "global": ns_global.score if ns_global is not None else None,
            "tree": ns_tree.score if ns_tree is not None else None,
            "node": ns_global,
        }

    def score_point(self, x: float, y: float, point_id: Optional[str] = None) -> dict:
        pid = point_id or f"pt_{x:.5f}_{y:.5f}"
        return self.score_sample(GeochemSample(site_code=pid, x=x, y=y))

    def score_points(self, xys: list[tuple[float, float]]) -> list[dict]:
        samples = [GeochemSample(site_code=f"pt_{x:.5f}_{y:.5f}", x=x, y=y)
                   for x, y in xys]
        self._register(samples)
        return [self.score_sample(s) for s in samples]


def load_fitted(path: str | Path) -> FittedModel:
    with open(Path(path), "rb") as f:
        bundle = pickle.load(f)
    if bundle.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported bundle version: {bundle.get('format_version')}")
    return FittedModel(bundle)
