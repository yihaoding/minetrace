"""ProspectivityModel — multi-source mineral prospectivity pipeline.

Data sources
------------
  Geochemistry : 5 assay types (stream sediment, rockchip, drillhole,
                 shallow drill, soil) — each yields neighbourhood statistics
  Geophysics   : magnetics, gravity, radiometrics, geochronology rasters
  Geology      : fault/worm proximity, geology unit, Cenozoic cover

Expert types (per geochemistry source)
---------------------------------------
  TargetEnrichmentExpert  — target element locally enriched above background?
  TargetPathfinderExpert  — pathfinder elements anomalous? (prior weights)
  ElementCorrelationExpert— element pairs co-enriched near deposits?  (new)

Geophysics experts (shared across all sources)
  MagneticExpert, GravityExpert, RadiometricExpert, GeochronExpert

Geology/structure experts
  FaultExpert, WormExpert, GeologyExpert

Weighting
---------
  Within each Expert: domain-knowledge weights + data-driven direction.
  Across Experts    : AUC-based weights in CompositeNode (one level only).

Usage::

    model = ProspectivityModel(catalog, config)
    model.plan(check_bbox=WA_BBOX)
    model.setup(exclude_ids=test_ids)
    scores = model.score_batch(samples)    # → list[NodeScore | None]
"""
from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.agent import AnalysisPlan
from core.catalog import DataCatalog, SourceSpec
from core.data_source import DataSourceRegistry
from core.expert_tree import CompositeNode, NodeScore
from domains.geochem.samples import GeochemInstance, GeochemSample

logger = logging.getLogger(__name__)

# WA bounding box (x_min, x_max, y_min, y_max)
WA_BBOX: Tuple[float, float, float, float] = (114.5, 128.5, -33.5, -15.0)

MIN_COVERAGE_FRACTION = 0.10
COVERAGE_PROBE_N      = 400  # 40 was too noisy: sources with true coverage
                             # near the 10% threshold flipped between runs
COVERAGE_MIN_SAMPLES  = 3
COVERAGE_RADIUS_KM    = 10.0


# ── Target configuration ──────────────────────────────────────────────────────

@dataclass
class TargetConfig:
    """All parameters that define one target-metal analysis run.

    Attributes
    ----------
    target         : element symbol, e.g. "Cu"
    pathfinders    : element → prior weight (domain knowledge)
    ratio_features : ratio feature names to include, e.g. ["ratio_Cu_Mo"]
    confidence_map : keyword → confidence weight for KB positive loading
    sites_csv      : path to known sites CSV
    layers         : expert layers to enable. Subset of {"geochem","geophys","structure"}.
                     Used for ablation studies. Default enables all three.
    """
    target: str
    pathfinders:    Dict[str, float] = field(default_factory=dict)
    ratio_features: List[str]        = field(default_factory=list)
    confidence_map: Dict[str, float] = field(default_factory=dict)
    sites_csv: str = ""
    layers: frozenset = field(default_factory=lambda: frozenset({"geochem","geophys","structure"}))


# ── PointEvidenceExpert ───────────────────────────────────────────────────────

class PointEvidenceExpert:
    """Direct measurement expert for individual-case mode.

    When a drillhole intercept or rock sample provides direct element
    concentrations at a point, this expert z-scores them against the
    background distribution and returns a score.
    """

    _invert_score: bool = False

    def __init__(
        self,
        name: str,
        evidence: Dict[str, float],
        background_mean: Dict[str, float],
        background_std:  Dict[str, float],
        sample_id: str,
    ) -> None:
        self.name = name
        self.required_sources: List[str] = ["__direct__"]
        self._evidence   = evidence
        self._bg_mean    = background_mean
        self._bg_std     = background_std
        self._sample_id  = sample_id
        self._fitted     = True

    def fit(self, positives, background) -> None:
        pass

    def fit_kb(self, background) -> None:
        pass

    def score_node(self, sample) -> Optional[NodeScore]:
        if sample.id != self._sample_id:
            return None
        total_s = total_w = 0.0
        z_scores: Dict[str, float] = {}
        for feat, val in self._evidence.items():
            if not np.isfinite(val):
                continue
            mu    = self._bg_mean.get(feat, 0.0)
            sigma = self._bg_std.get(feat, 1.0)
            if sigma < 1e-9:
                sigma = 1.0
            z = (val - mu) / sigma
            z_scores[feat] = float(z)
            total_s += z
            total_w += 1.0
        if total_w < 1e-9:
            return None
        raw = total_s / total_w
        if self._invert_score:
            raw = -raw
            z_scores = {f: -z for f, z in z_scores.items()}
        feature_w = {f: 1.0 / total_w for f in z_scores}
        return NodeScore(
            name=self.name,
            score=float(1.0 / (1.0 + np.exp(-raw))),
            weights=dict(feature_w),
            feature_z=z_scores,
            feature_w=feature_w,
        )

    def score(self, sample) -> Optional[float]:
        ns = self.score_node(sample)
        return ns.score if ns is not None else None


# ── ProspectivityModel ────────────────────────────────────────────────────────

class ProspectivityModel:
    """Multi-source mineral prospectivity model.

    Layers
    ------
    1. DataCatalog   — describes available data sources
    2. plan()        — coverage probe → decide which sources to activate
    3. setup()       — load sources, generate background, build + fit experts
    4. score_batch() — register new samples, return NodeScore tree per sample
    """

    def __init__(
        self,
        catalog: DataCatalog,
        config: Optional[TargetConfig] = None,
        n_bg: int = 300,
        bbox: Tuple[float, float, float, float] = WA_BBOX,
        seed: int = 42,
        *,
        target_config: Optional[TargetConfig] = None,  # backward-compat alias
    ) -> None:
        if config is None and target_config is not None:
            config = target_config
        if config is None:
            raise ValueError("config (or target_config) must be provided")
        self._catalog  = catalog
        self._config   = config
        self._n_bg     = n_bg
        self._bbox     = bbox
        self._seed     = seed
        self._rng      = np.random.default_rng(seed)
        self._plan:    Optional[AnalysisPlan] = None
        self._tree:    Optional[CompositeNode] = None
        self._active_source_names: List[str] = []
        self._active_expert_names: List[str] = []
        self._bg_element_mean: Dict[str, float] = {}
        self._bg_element_std:  Dict[str, float] = {}
        self._registry = DataSourceRegistry
        self._fitted   = False

    # ── public interface ──────────────────────────────────────────────────────

    def plan(
        self,
        check_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> AnalysisPlan:
        """Decide which sources to activate based on coverage probing."""
        active: List[str] = []
        rationale: Dict[str, str] = {}
        bbox = check_bbox if check_bbox is not None else self._bbox

        for spec in self._catalog.all():
            if spec.source_type == "assay_spatial":
                if check_bbox is not None:
                    frac = self._probe_coverage(spec, bbox)
                    if frac < MIN_COVERAGE_FRACTION:
                        rationale[spec.name] = (
                            f"skipped: coverage {frac:.1%} < {MIN_COVERAGE_FRACTION:.0%}")
                    else:
                        active.append(spec.name)
                        rationale[spec.name] = f"activated: coverage {frac:.1%}"
                else:
                    active.append(spec.name)
                    rationale[spec.name] = "activated: trust_catalog"
            elif spec.source_type in ("raster", "vector"):
                active.append(spec.name)
                rationale[spec.name] = f"activated: {spec.source_type}"
            else:
                rationale[spec.name] = f"skipped: unknown source_type '{spec.source_type}'"

        expert_descs = self._describe_experts(active)
        self._plan = AnalysisPlan(
            active_source_names=active,
            expert_tree_description=expert_descs,
            rationale=rationale,
        )
        return self._plan

    def setup(self, exclude_ids: Optional[set] = None) -> None:
        """Load sources, generate background, build and fit expert tree.

        Parameters
        ----------
        exclude_ids : SITE_CODE strings to withhold from training (held-out eval).
        """
        if self._plan is None:
            self.plan()
        self._registry.clear()

        active = self._plan.active_source_names
        kb_instances: List[GeochemInstance] = []

        if self._config.sites_csv:
            kb_instances, _ = self._load_kb(self._config.sites_csv)
            if exclude_ids:
                before = len(kb_instances)
                kb_instances = [i for i in kb_instances
                                if i.sample.id not in exclude_ids]
                logger.info(
                    "ProspectivityModel: excluded %d held-out sites; %d KB remain",
                    before - len(kb_instances), len(kb_instances),
                )
        else:
            logger.warning("ProspectivityModel: no sites_csv provided.")

        assay_sources, geo_source, struct_source = [], None, None

        for name in active:
            spec = self._catalog.get(name)
            if spec.source_type == "assay_spatial":
                src = self._load_assay_source(spec)
                if src is not None:
                    self._registry.register(src)
                    assay_sources.append(src)
            elif spec.source_type == "raster" and spec.modality == "geophysics":
                if geo_source is None:
                    geo_source = self._load_geophysics_source(active)
                    if geo_source is not None:
                        self._registry.register(geo_source)
            elif spec.source_type == "vector" and spec.modality == "geology":
                if struct_source is None:
                    struct_source = self._load_structure_source(active)
                    if struct_source is not None:
                        self._registry.register(struct_source)

        pos_samples = [inst.sample for inst in kb_instances]
        background  = self._generate_background(assay_sources)
        logger.info("ProspectivityModel: %d background samples", len(background))

        all_samples = pos_samples + background
        for src in assay_sources:
            src.register_samples(all_samples)
        if geo_source is not None:
            geo_source.register_samples(all_samples)
        if struct_source is not None:
            struct_source.register_samples(all_samples)

        self._fit_bg_element_stats(background, assay_sources)

        # ── build experts ──────────────────────────────────────────────────
        # 3 multi-source geochem experts + 4 geophysics + 3 geology = 10 fixed
        experts = []
        target       = self._config.target
        assay_names  = [s.name for s in assay_sources]

        layers = self._config.layers

        if assay_names and "geochem" in layers:
            from domains.geochem.experts.geochem_experts import (
                TargetEnrichmentExpert,
                TargetPathfinderExpert,
                ElementCorrelationExpert,
            )
            # Co-enrichment pairs: target + all pathfinders
            pf_elems   = [target] + list(self._config.pathfinders.keys())
            corr_pairs = [
                (pf_elems[i], pf_elems[j])
                for i in range(len(pf_elems))
                for j in range(i + 1, len(pf_elems))
            ]
            experts.append(TargetEnrichmentExpert(target, assay_names))
            experts.append(TargetPathfinderExpert(
                target, assay_names,
                self._config.pathfinders,
                self._config.ratio_features,
            ))
            experts.append(ElementCorrelationExpert(target, assay_names, corr_pairs))

        if geo_source is not None and "geophys" in layers:
            from domains.geochem.experts.geophysics_experts import (
                MagneticExpert, GravityExpert, RadiometricExpert, GeochronExpert,
            )
            experts.extend([
                MagneticExpert(geo_source.name),
                GravityExpert(geo_source.name),
                RadiometricExpert(geo_source.name),
                GeochronExpert(geo_source.name),
            ])

        if struct_source is not None and "structure" in layers:
            from domains.geochem.experts.structure_experts import (
                FaultExpert, WormExpert, GeologyExpert,
            )
            experts.extend([
                FaultExpert(struct_source.name),
                WormExpert(struct_source.name),
                GeologyExpert(struct_source.name),
            ])

        self._tree = CompositeNode(
            name=f"{target.lower()}_prospectivity",
            children=experts,
        )
        logger.info(
            "ProspectivityModel: fitting %d experts, %d positives, %d background",
            len(experts), len(pos_samples), len(background),
        )
        self._tree.fit(pos_samples, background)

        self._active_source_names = (
            [s.name for s in assay_sources]
            + ([geo_source.name]    if geo_source    is not None else [])
            + ([struct_source.name] if struct_source is not None else [])
        )
        self._active_expert_names = [e.name for e in experts]
        # Retain the fit sets so post-hoc scorers (e.g. belief fusion) can
        # calibrate per-expert reliability without re-deriving them.
        self._fit_positives = pos_samples
        self._fit_background = background
        self._fitted = True
        logger.info("ProspectivityModel: ready.")

    def score_batch(self, samples: List[GeochemSample]) -> list:
        if self._tree is None:
            raise RuntimeError("Call setup() first.")
        for name in self._active_source_names:
            try:
                src = self._registry.get(name)
                if hasattr(src, "register_samples"):
                    src.register_samples(samples)
            except KeyError:
                pass
        return [self._tree.score_node(s) for s in samples]

    def score_case(
        self,
        sample: GeochemSample,
        direct_evidence: Optional[Dict[str, float]] = None,
    ) -> Optional[NodeScore]:
        """Score with optional direct measurements (individual case mode)."""
        if self._tree is None:
            raise RuntimeError("Call setup() first.")
        for name in self._active_source_names:
            try:
                src = self._registry.get(name)
                if hasattr(src, "register_samples"):
                    src.register_samples([sample])
            except KeyError:
                pass
        ns_base = self._tree.score_node(sample)
        if not direct_evidence:
            return ns_base

        ev_expert = PointEvidenceExpert(
            name=f"direct_evidence_{sample.id}",
            evidence=direct_evidence,
            background_mean=self._bg_element_mean,
            background_std=self._bg_element_std,
            sample_id=sample.id,
        )
        ns_ev = ev_expert.score_node(sample)
        if ns_base is None and ns_ev is None:
            return None
        children: Dict[str, NodeScore] = {}
        vals = []
        if ns_base is not None:
            children[ns_base.name] = ns_base; vals.append(ns_base.score)
        if ns_ev is not None:
            children[ns_ev.name]   = ns_ev;   vals.append(ns_ev.score)
        return NodeScore(
            name=f"{self._config.target.lower()}_prospectivity_case",
            score=float(np.mean(vals)),
            children=children,
            weights={n: 1.0 / len(children) for n in children},
        )

    def active_sources(self) -> List[str]:
        return list(self._active_source_names)

    def active_experts(self) -> List[str]:
        return list(self._active_expert_names)

    # ── internal ──────────────────────────────────────────────────────────────

    def _probe_coverage(self, spec: SourceSpec, bbox) -> float:
        try:
            src = self._load_assay_source(spec)
            if src is None:
                return 0.0
            x_min, x_max, y_min, y_max = bbox
            # zlib.crc32 (not hash()): str hash is salted per process, which
            # made borderline sources flip active/skipped between runs.
            rng = np.random.default_rng(self._seed + zlib.crc32(spec.name.encode()) % 10000)
            xs  = rng.uniform(x_min, x_max, COVERAGE_PROBE_N)
            ys  = rng.uniform(y_min, y_max, COVERAGE_PROBE_N)
            hits = sum(
                1 for x, y in zip(xs, ys)
                if src.has_coverage(x, y, COVERAGE_MIN_SAMPLES, COVERAGE_RADIUS_KM)
            )
            return hits / COVERAGE_PROBE_N
        except Exception as exc:
            logger.warning("Coverage probe failed for %s: %s", spec.name, exc)
            return 0.0

    def _load_assay_source(self, spec: SourceSpec):
        from domains.geochem.sources.assay import AssaySource
        if not spec.path:
            return None
        filter_val = spec.loader_kwargs.get("sample_type_filter")
        try:
            return AssaySource(
                csv_path=spec.path, name=spec.name,
                sample_type_filter=filter_val,
                center_lat=spec.loader_kwargs.get("center_lat", -25.0),
            )
        except Exception as exc:
            logger.error("Failed to load AssaySource '%s': %s", spec.name, exc)
            return None

    def _load_geophysics_source(self, active_names: List[str]):
        from domains.geochem.sources.geophysics import GeophysicsSource
        raster_paths: Dict[str, str] = {}
        for name in active_names:
            spec = self._catalog.get(name)
            if spec.source_type == "raster" and spec.modality == "geophysics":
                rkey = spec.loader_kwargs.get("raster_key", spec.subtype or spec.name)
                if spec.path:
                    raster_paths[rkey] = spec.path
        if not raster_paths:
            return None
        try:
            return GeophysicsSource(raster_paths=raster_paths, name="geophysics")
        except Exception as exc:
            logger.error("Failed to load GeophysicsSource: %s", exc)
            return None

    def _load_structure_source(self, active_names: List[str]):
        from domains.geochem.sources.geology import StructureSource
        paths: Dict[str, str] = {}
        for name in active_names:
            spec = self._catalog.get(name)
            if spec.source_type == "vector" and spec.modality == "geology":
                lkey = spec.loader_kwargs.get("layer_key", spec.subtype or spec.name)
                if spec.path:
                    paths[lkey] = spec.path
        required = {"fault", "worm_mag", "worm_grav", "geology", "cenozoic"}
        if required - set(paths.keys()):
            logger.warning("StructureSource: missing layers %s", required - set(paths.keys()))
            return None
        try:
            return StructureSource(
                fault_path=paths["fault"], worm_mag_path=paths["worm_mag"],
                worm_grav_path=paths["worm_grav"], geology_path=paths["geology"],
                cenozoic_path=paths["cenozoic"], name="structure",
            )
        except Exception as exc:
            logger.error("Failed to load StructureSource: %s", exc)
            return None

    def _load_kb(self, sites_csv: str):
        from domains.geochem.sources.site_catalog import SiteCatalogSource
        import pandas as pd
        try:
            catalog_src = SiteCatalogSource(sites_csv)
            all_samples = catalog_src.load_all_samples()
            df = pd.read_csv(sites_csv)
            target_col = f"{self._config.target}_SITES"
            instances = []
            if target_col in df.columns:
                for _, row in df.iterrows():
                    val = str(row.get(target_col, ""))
                    conf = max(
                        (c for kw, c in self._config.confidence_map.items() if kw in val),
                        default=0.0,
                    )
                    if conf > 0:
                        sid = str(row["SITE_CODE"])
                        commo = [c.strip() for c in str(row.get("SITE_COMMO", "")).split(",") if c.strip()]
                        s = GeochemSample(
                            site_code=sid, x=float(row["X"]), y=float(row["Y"]),
                            site_commo=commo,
                            site_stage=str(row.get("SITE_STAGE", "")),
                            site_type=str(row.get("SITE_TYPE_", "")),
                        )
                        instances.append(GeochemInstance(
                            sample=s, confidence=conf,
                            metadata={target_col: val},
                        ))
            else:
                instances = catalog_src.load_cu_instances()
            return instances, all_samples
        except Exception as exc:
            logger.error("Failed to load KB from '%s': %s", sites_csv, exc)
            return [], []

    def _generate_background(self, assay_sources: list) -> List[GeochemSample]:
        x_min, x_max, y_min, y_max = self._bbox
        background: List[GeochemSample] = []
        attempts = 0
        while len(background) < self._n_bg and attempts < self._n_bg * 200:
            attempts += 1
            x = float(self._rng.uniform(x_min, x_max))
            y = float(self._rng.uniform(y_min, y_max))
            if assay_sources and not any(
                s.has_coverage(x, y, COVERAGE_MIN_SAMPLES, COVERAGE_RADIUS_KM)
                for s in assay_sources
            ):
                continue
            background.append(GeochemSample(
                site_code=f"BG_{len(background):04d}", x=x, y=y,
            ))
        return background

    def _fit_bg_element_stats(self, background: List[GeochemSample], assay_sources: list) -> None:
        if not assay_sources or not background:
            return
        from domains.geochem.sources.assay import DEFAULT_ELEMENTS
        element_vals: Dict[str, List[float]] = {e: [] for e in DEFAULT_ELEMENTS}
        src = assay_sources[0]
        for s in background:
            fv = src.compute_for_point(s.x, s.y)
            for elem in DEFAULT_ELEMENTS:
                v = fv.get(f"{elem}_10km_median", np.nan)
                if np.isfinite(v):
                    element_vals[elem].append(v)
        for elem, vals in element_vals.items():
            key = f"{elem}_ppm"
            arr = np.array(vals)
            self._bg_element_mean[key] = float(np.nanmean(arr)) if len(arr) > 0 else 0.0
            self._bg_element_std[key]  = float(np.nanstd(arr))  if len(arr) > 1 else 1.0
            if self._bg_element_std[key] < 1e-9:
                self._bg_element_std[key] = 1.0

    def _describe_experts(self, active_names: List[str]) -> List[str]:
        target      = self._config.target
        assay_names = [n for n in active_names
                       if self._catalog.get(n).source_type == "assay_spatial"]
        descs = []
        if assay_names:
            descs += [
                f"TargetEnrichmentExpert({target}, sources={assay_names})",
                f"TargetPathfinderExpert({target}, pf={list(self._config.pathfinders)}, sources={assay_names})",
                f"ElementCorrelationExpert({target}, sources={assay_names})",
            ]
        if any(self._catalog.get(n).source_type == "raster" for n in active_names):
            descs += ["MagneticExpert", "GravityExpert", "RadiometricExpert", "GeochronExpert"]
        if any(self._catalog.get(n).source_type == "vector" for n in active_names):
            descs += ["FaultExpert", "WormExpert", "GeologyExpert"]
        return descs


# ── backward-compatibility aliases ───────────────────────────────────────────

GeochemAgent       = ProspectivityModel
GeochemTargetConfig = TargetConfig
