"""run_phase3_all_features.py

Phase 3 comparison: original experts vs. full-feature experts on identical test sets.

Original experts (Phase 3 baseline):
  TargetEnrichmentExpert  — 6 features: contrast_5_50/10_50, p90, frac_above at 5/10 km
  TargetPathfinderExpert  — 4 stats × n_pathfinders + ratio features

Full-feature experts (this script):
  FullEnrichmentExpert    — 10 features: adds median + cv at 5/10 km to original
  FullPathfinderExpert    — 10 stats × n_pathfinders + ratio features (adds median + cv)
  PCAExpert               — PC1–5 median at 5 km and 10 km (10 features, equal weight)
  CoherenceExpert         — spatial_coherence_10km (negated: lower = more clustered = anomalous)

Usage:
    cd /group/pmc050/yding/gad_reasoning
    python scripts/run_phase3_all_features.py

Output:
    reports/phase3_all_features.md
    reports/phase3_all_features.png
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_source import DataSourceRegistry
from core.expert import ExpertRegistry
from core.pipeline import L1Pipeline
from domains.geochem.background import ExcludeInstancesBackground
from domains.geochem.evaluation import compute_auc, expert_auc, top_k_precision
from domains.geochem.evaluation import WA_X_MIN, WA_X_MAX, WA_Y_MIN, WA_Y_MAX
from domains.geochem.experts.generic_target import TargetEnrichmentExpert, TargetPathfinderExpert
from domains.geochem.samples import GeochemSample, GeochemInstance
from domains.geochem.sources.sediment_source import SedimentSource

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
SITES_DIR = ROOT / "datasets/geochemical/states/sites"
GEO_DIR = ROOT / "datasets/geochemical/states/state_geochemical"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

SED_CSV = GEO_DIR / "gswa_all_sediment.csv"
ROCK_CSV = GEO_DIR / "gswa_all_rockchips.csv"

# ── target config (same as phase3) ───────────────────────────────────────────
TARGETS: dict[str, dict] = {
    "Cu": {
        "sites_csv": SITES_DIR / "gswa_cu_sites.csv",
        "site_col": "Cu_SITES",
        "confidence_map": {"Mine": 1.0, "Deposit": 1.0},
        "pathfinders": {"Au": 2.0, "Mo": 1.5, "Ag": 1.0, "Co": 0.8, "Bi": 0.7},
        "ratio_features": ["ratio_Cu_Mo", "ratio_Cu_Pb", "ratio_Co_Ni"],
    },
    "Au": {
        "sites_csv": SITES_DIR / "gswa_au_site.csv",
        "site_col": "Au_SITES",
        "confidence_map": {"Deposit": 1.0, "Prospect": 0.5},
        "pathfinders": {"Ag": 1.5, "As": 1.0, "Bi": 0.8, "Mo": 0.6},
        "ratio_features": ["ratio_Au_Cu", "ratio_Au_As"],
    },
    "W": {
        "sites_csv": SITES_DIR / "gswa_w_site.csv",
        "site_col": "W_SITES",
        "confidence_map": {"Mine": 1.0, "Deposit": 1.0, "Prospect": 0.5},
        "pathfinders": {"Sn": 1.5, "Mo": 1.2, "Bi": 1.0, "As": 0.8},
        "ratio_features": ["ratio_W_Sn", "ratio_W_Mo"],
    },
    "Ni": {
        "sites_csv": SITES_DIR / "gswa_ni_site.csv",
        "site_col": "Ni_SITES",
        "confidence_map": {"Mine": 1.0, "Deposit": 1.0, "Prospect": 0.5},
        "pathfinders": {"Co": 2.0, "Cr": 1.5, "Cu": 0.8},
        "ratio_features": ["ratio_Ni_Co", "ratio_Ni_Cr"],
    },
}

N_POS_TEST_FRAC = 0.10
N_POS_TEST_MIN = 30
N_TRAIN_BG = 300
MIN_SED_COVERAGE = 3


# ── New Expert classes ────────────────────────────────────────────────────────

class FullEnrichmentExpert:
    """Target-element enrichment using all 10 local stats.

    Adds median and cv (at 5 km and 10 km) to the 6 features used by
    TargetEnrichmentExpert.

    Features (10 total):
      {target}_{5/10}km_median   — absolute concentration level (log-space)
      {target}_{5/10}km_p90      — upper-tail intensity
      {target}_{5/10}km_cv       — within-neighbourhood variability (hotspot signal)
      {target}_{5/10}km_frac_above — fraction exceeding regional 75th percentile
      {target}_contrast_{5/10}_50  — local vs 50 km background enrichment
    """

    def __init__(self, target: str, source_name: str) -> None:
        self.name = f"{target.lower()}_{source_name}_full_enrichment"
        self.required_sources: list[str] = [source_name]
        self._source_name = source_name
        self._features = [
            f"{target}_5km_median",     f"{target}_10km_median",
            f"{target}_5km_p90",        f"{target}_10km_p90",
            f"{target}_5km_cv",         f"{target}_10km_cv",
            f"{target}_5km_frac_above", f"{target}_10km_frac_above",
            f"{target}_contrast_5_50",  f"{target}_contrast_10_50",
        ]
        self._bg_mean: dict[str, float] = {}
        self._bg_std: dict[str, float] = {}
        self._fitted = False

    def fit_kb(self, background_samples: list) -> None:
        src = DataSourceRegistry.get(self._source_name)
        values: dict[str, list[float]] = {f: [] for f in self._features}
        for s in background_samples:
            fv = src.get_features(s.id)
            if fv is None:
                continue
            for f in self._features:
                v = fv.get(f, np.nan)
                if np.isfinite(v):
                    values[f].append(v)
        for f in self._features:
            arr = np.array(values[f])
            self._bg_mean[f] = float(arr.mean()) if len(arr) > 0 else 0.0
            self._bg_std[f] = float(arr.std()) if len(arr) > 1 else 1.0
            if self._bg_std[f] < 1e-9:
                self._bg_std[f] = 1.0
        self._fitted = True

    def score(self, sample) -> float | None:
        if not self._fitted:
            return None
        src = DataSourceRegistry.get(self._source_name)
        fv = src.get_features(sample.id)
        if fv is None:
            return None
        z_scores = []
        for f in self._features:
            v = fv.get(f, np.nan)
            if np.isfinite(v):
                z_scores.append((v - self._bg_mean[f]) / self._bg_std[f])
        if not z_scores:
            return None
        return float(1.0 / (1.0 + np.exp(-float(np.mean(z_scores)))))


class FullPathfinderExpert:
    """Pathfinder enrichment using all 10 stats per element.

    Extends TargetPathfinderExpert by adding median and cv to the per-element
    feature set (was 4 stats, now 10).

    Per-element features (10 × n_pathfinders):
      {elem}_{5/10}km_median, p90, cv, frac_above, contrast_5_50, contrast_10_50
    Plus ratio features at weight 1.0.
    """

    def __init__(
        self,
        target: str,
        source_name: str,
        pathfinders: dict[str, float],
        ratio_features: list[str] | None = None,
    ) -> None:
        self.name = f"{target.lower()}_{source_name}_full_pathfinder"
        self.required_sources: list[str] = [source_name]
        self._source_name = source_name
        self._pathfinders = pathfinders
        self._ratio_features = ratio_features or []
        self._feature_weights: list[tuple[str, float]] = []
        self._bg_mean: dict[str, float] = {}
        self._bg_std: dict[str, float] = {}
        self._fitted = False

    def fit_kb(self, background_samples: list) -> None:
        src = DataSourceRegistry.get(self._source_name)
        sample_fv = None
        for s in background_samples:
            sample_fv = src.get_features(s.id)
            if sample_fv is not None:
                break

        self._feature_weights = []
        pf_stats = [
            "5km_median",     "10km_median",
            "5km_p90",        "10km_p90",
            "5km_cv",         "10km_cv",
            "5km_frac_above", "10km_frac_above",
            "contrast_5_50",  "contrast_10_50",
        ]
        for elem, w in self._pathfinders.items():
            for stat in pf_stats:
                fname = f"{elem}_{stat}"
                if sample_fv is None or fname in sample_fv:
                    self._feature_weights.append((fname, w))
        for rf in self._ratio_features:
            if sample_fv is None or rf in sample_fv:
                self._feature_weights.append((rf, 1.0))

        values: dict[str, list[float]] = {f: [] for f, _ in self._feature_weights}
        for s in background_samples:
            fv = src.get_features(s.id)
            if fv is None:
                continue
            for f, _ in self._feature_weights:
                v = fv.get(f, np.nan)
                if np.isfinite(v):
                    values[f].append(v)
        for f, _ in self._feature_weights:
            arr = np.array(values[f])
            self._bg_mean[f] = float(arr.mean()) if len(arr) > 0 else 0.0
            self._bg_std[f] = float(arr.std()) if len(arr) > 1 else 1.0
            if self._bg_std[f] < 1e-9:
                self._bg_std[f] = 1.0
        self._fitted = True

    def score(self, sample) -> float | None:
        if not self._fitted or not self._feature_weights:
            return None
        src = DataSourceRegistry.get(self._source_name)
        fv = src.get_features(sample.id)
        if fv is None:
            return None
        weighted_z, total_w = 0.0, 0.0
        for f, w in self._feature_weights:
            v = fv.get(f, np.nan)
            if np.isfinite(v):
                z = (v - self._bg_mean[f]) / self._bg_std[f]
                weighted_z += w * z
                total_w += w
        if total_w < 1e-9:
            return None
        return float(1.0 / (1.0 + np.exp(-weighted_z / total_w)))


class PCAExpert:
    """Anomaly signal from the global PCA neighbourhood scores.

    Uses PC1–5 median at 5 km and 10 km (10 features, equal weight).
    The global PCA is fitted on all assay samples in SedimentSource, so each
    PC captures a dominant geochemical co-variance pattern.  A site whose
    neighbourhood PCA profile deviates from the background distribution
    (in any PC direction) scores high.

    Note: direction of anomaly per PC is learned from the background
    distribution via z-score normalisation, so no manual sign flip needed.
    """

    def __init__(self, source_name: str) -> None:
        self.name = f"{source_name}_pca"
        self.required_sources: list[str] = [source_name]
        self._source_name = source_name
        self._features = [
            f"PC{i}_{scale}km_median"
            for scale in [5, 10]
            for i in range(1, 6)
        ]
        self._bg_mean: dict[str, float] = {}
        self._bg_std: dict[str, float] = {}
        self._fitted = False

    def fit_kb(self, background_samples: list) -> None:
        src = DataSourceRegistry.get(self._source_name)
        values: dict[str, list[float]] = {f: [] for f in self._features}
        for s in background_samples:
            fv = src.get_features(s.id)
            if fv is None:
                continue
            for f in self._features:
                v = fv.get(f, np.nan)
                if np.isfinite(v):
                    values[f].append(v)
        for f in self._features:
            arr = np.array(values[f])
            self._bg_mean[f] = float(arr.mean()) if len(arr) > 0 else 0.0
            self._bg_std[f] = float(arr.std()) if len(arr) > 1 else 1.0
            if self._bg_std[f] < 1e-9:
                self._bg_std[f] = 1.0
        self._fitted = True

    def score(self, sample) -> float | None:
        if not self._fitted:
            return None
        src = DataSourceRegistry.get(self._source_name)
        fv = src.get_features(sample.id)
        if fv is None:
            return None
        z_scores = []
        for f in self._features:
            v = fv.get(f, np.nan)
            if np.isfinite(v):
                z_scores.append((v - self._bg_mean[f]) / self._bg_std[f])
        if not z_scores:
            return None
        return float(1.0 / (1.0 + np.exp(-float(np.mean(z_scores)))))


class CoherenceExpert:
    """Spatial clustering signal from spatial_coherence_10km.

    spatial_coherence_10km = mean pairwise distance (in degrees) of the
    top-5 primary-element samples within 10 km.

    NEGATED in scoring: lower distance = more spatially clustered = more
    anomalous.  The z-score is therefore flipped before the sigmoid.

    Caveat: the primary element is always self._elements[0] (Cu) regardless
    of the target commodity — this expert is most meaningful for Cu targeting.
    """

    def __init__(self, source_name: str) -> None:
        self.name = f"{source_name}_coherence"
        self.required_sources: list[str] = [source_name]
        self._source_name = source_name
        self._bg_mean: float = 0.0
        self._bg_std: float = 1.0
        self._fitted = False

    def fit_kb(self, background_samples: list) -> None:
        src = DataSourceRegistry.get(self._source_name)
        values: list[float] = []
        for s in background_samples:
            fv = src.get_features(s.id)
            if fv is None:
                continue
            v = fv.get("spatial_coherence_10km", np.nan)
            if np.isfinite(v):
                values.append(v)
        arr = np.array(values)
        self._bg_mean = float(arr.mean()) if len(arr) > 0 else 0.0
        self._bg_std = float(arr.std()) if len(arr) > 1 else 1.0
        if self._bg_std < 1e-9:
            self._bg_std = 1.0
        self._fitted = True

    def score(self, sample) -> float | None:
        if not self._fitted:
            return None
        src = DataSourceRegistry.get(self._source_name)
        fv = src.get_features(sample.id)
        if fv is None:
            return None
        v = fv.get("spatial_coherence_10km", np.nan)
        if not np.isfinite(v):
            return None
        z = (v - self._bg_mean) / self._bg_std
        # Negate: lower coherence (more clustered) → higher anomaly score
        return float(1.0 / (1.0 + np.exp(z)))


# ── helpers (from run_phase3_multitarget.py) ──────────────────────────────────

def load_instances(
    sites_csv: Path,
    elem_col: str,
    confidence_map: dict[str, float] | None = None,
) -> tuple[list[GeochemInstance], list[GeochemInstance]]:
    if confidence_map is None:
        confidence_map = {"Mine": 1.0, "Deposit": 1.0}
    df = pd.read_csv(sites_csv)
    high_conf, weak = [], []
    for _, row in df.iterrows():
        val = str(row.get(elem_col, ""))
        conf = 0.0
        for keyword, c in confidence_map.items():
            if keyword in val:
                conf = max(conf, c)
        if conf == 0.0:
            continue
        site_code = str(row.get("SITE_CODE", row.get("fid", "")))
        try:
            x, y = float(row["X"]), float(row["Y"])
        except (ValueError, KeyError):
            continue
        s = GeochemSample(
            site_code=site_code, x=x, y=y,
            site_stage=str(row.get("SITE_STAGE", "")),
            site_type=str(row.get("SITE_TYPE_", "")),
        )
        inst = GeochemInstance(sample=s, confidence=conf, metadata={"val": val})
        if conf >= 1.0:
            high_conf.append(inst)
        else:
            weak.append(inst)
    return high_conf, weak


def build_test_set(
    high_conf: list[GeochemInstance],
    weak_instances: list[GeochemInstance],
    sed_src: SedimentSource,
    seed: int,
) -> tuple[list[GeochemSample], list[int], list[GeochemInstance]]:
    rng = np.random.default_rng(seed)
    n_test = max(N_POS_TEST_MIN, int(len(high_conf) * N_POS_TEST_FRAC))
    n_test = min(n_test, len(high_conf))

    neg: list[GeochemSample] = []
    attempts = 0
    while len(neg) < n_test and attempts < n_test * 300:
        attempts += 1
        x = float(rng.uniform(WA_X_MIN, WA_X_MAX))
        y = float(rng.uniform(WA_Y_MIN, WA_Y_MAX))
        if sed_src.compute_for_point(x, y).get("n_samples_10km", 0) < MIN_SED_COVERAGE:
            continue
        neg.append(GeochemSample(site_code=f"NEG_{len(neg):04d}", x=x, y=y))
    sed_src.register_samples(neg)

    test_idx = rng.choice(len(high_conf), size=n_test, replace=False)
    test_pos_ids = {high_conf[i].sample.id for i in test_idx}
    test_pos = [high_conf[i].sample for i in test_idx]
    sed_src.register_samples(test_pos)

    train = [inst for inst in high_conf if inst.sample.id not in test_pos_ids] + weak_instances
    logger.info("%d pos test | %d neg test | %d train KB", len(test_pos), len(neg), len(train))
    return test_pos + neg, [1] * len(test_pos) + [0] * len(neg), train


def generate_train_bg(
    sed_src: SedimentSource,
    n: int,
    seed: int,
) -> list[GeochemSample]:
    rng = np.random.default_rng(seed)
    samples: list[GeochemSample] = []
    attempts = 0
    while len(samples) < n and attempts < n * 300:
        attempts += 1
        x = float(rng.uniform(WA_X_MIN, WA_X_MAX))
        y = float(rng.uniform(WA_Y_MIN, WA_Y_MAX))
        if sed_src.compute_for_point(x, y).get("n_samples_10km", 0) < MIN_SED_COVERAGE:
            continue
        samples.append(GeochemSample(site_code=f"TRBG_{len(samples):04d}", x=x, y=y))
    sed_src.register_samples(samples)
    return samples


def run_pipeline(
    config_label: str,
    register_fn,
    all_train: list[GeochemSample],
    train_instances: list[GeochemInstance],
    test_samples: list[GeochemSample],
    test_labels: list[int],
) -> dict:
    """Register experts via register_fn, fit pipeline, score test set, return metrics."""
    ExpertRegistry.clear()
    register_fn()
    bg_fn = ExcludeInstancesBackground()
    pipeline = L1Pipeline()
    pipeline.fit(all_train, train_instances, background_fn=bg_fn)
    results = pipeline.score_all(test_samples)
    auc = compute_auc(results, test_labels)
    expert_aucs = {
        name: expert_auc(results, test_labels, name)
        for name in pipeline.active_expert_names
    }
    prec20 = top_k_precision(results, test_labels, min(20, len(test_labels) // 2))
    prec40 = top_k_precision(results, test_labels, min(40, len(test_labels)))
    logger.info("[%s] AUC=%.3f  P@20=%.3f  P@40=%.3f", config_label, auc, prec20, prec40)
    return {
        "auc": auc, "prec20": prec20, "prec40": prec40,
        "expert_aucs": expert_aucs,
        "active_experts": list(pipeline.active_expert_names),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("Loading sediment source …")
    sed_src = SedimentSource(SED_CSV, name="sediment")
    logger.info("Loading rockchips source …")
    rock_src = SedimentSource(ROCK_CSV, name="rockchips")

    DataSourceRegistry.register(sed_src)
    DataSourceRegistry.register(rock_src)

    logger.info("Generating shared training background …")
    train_bg = generate_train_bg(sed_src, N_TRAIN_BG, seed=99)
    rock_src.register_samples(train_bg)

    all_results: dict[str, dict] = {}

    for target, cfg in TARGETS.items():
        logger.info("=" * 60)
        logger.info("TARGET: %s", target)

        high_conf, weak = load_instances(cfg["sites_csv"], cfg["site_col"], cfg.get("confidence_map"))
        test_samples, test_labels, train_instances = build_test_set(
            high_conf, weak, sed_src, seed=42,
        )
        all_target_samples = test_samples + [inst.sample for inst in train_instances]
        rock_src.register_samples(all_target_samples)

        train_positive_samples = [inst.sample for inst in train_instances]
        all_train = train_positive_samples + train_bg

        # ── ORIGINAL experts ──────────────────────────────────────────────
        def register_original():
            for src_name in ("sediment", "rockchips"):
                ExpertRegistry.register(TargetEnrichmentExpert(target, src_name))
                ExpertRegistry.register(TargetPathfinderExpert(
                    target, src_name, cfg["pathfinders"],
                    ratio_features=cfg.get("ratio_features", []),
                ))

        orig = run_pipeline(
            f"{target}/original", register_original,
            all_train, train_instances, test_samples, test_labels,
        )

        # ── FULL-FEATURE experts ──────────────────────────────────────────
        def register_full():
            for src_name in ("sediment", "rockchips"):
                ExpertRegistry.register(FullEnrichmentExpert(target, src_name))
                ExpertRegistry.register(FullPathfinderExpert(
                    target, src_name, cfg["pathfinders"],
                    ratio_features=cfg.get("ratio_features", []),
                ))
                ExpertRegistry.register(PCAExpert(src_name))
                ExpertRegistry.register(CoherenceExpert(src_name))

        full = run_pipeline(
            f"{target}/all_features", register_full,
            all_train, train_instances, test_samples, test_labels,
        )

        all_results[target] = {
            "orig": orig,
            "full": full,
            "n_test_pos": sum(test_labels),
            "n_test_neg": len(test_labels) - sum(test_labels),
        }

    # ── print comparison table ────────────────────────────────────────────────
    print("\n========== FEATURE COVERAGE COMPARISON ==========")
    print(f"{'Target':<6}  {'Orig AUC':>9}  {'Full AUC':>9}  {'Δ AUC':>7}  {'P@20 orig':>9}  {'P@20 full':>9}")
    print("-" * 58)
    for target, r in all_results.items():
        delta = r["full"]["auc"] - r["orig"]["auc"]
        sign = "+" if delta >= 0 else ""
        print(
            f"{target:<6}  {r['orig']['auc']:>9.3f}  {r['full']['auc']:>9.3f}"
            f"  {sign}{delta:>6.3f}  {r['orig']['prec20']:>9.3f}  {r['full']['prec20']:>9.3f}"
        )
    print("=" * 58)

    # ── per-expert AUC breakdown for full config ──────────────────────────────
    print("\nPer-expert AUC (full-feature config):")
    for target, r in all_results.items():
        print(f"\n  {target}:")
        for name, val in sorted(r["full"]["expert_aucs"].items()):
            src = "sed" if "sediment" in name else "rock"
            print(f"    [{src}] {name}: {val:.3f}")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (target, r) in zip(axes, all_results.items()):
        expert_aucs = r["full"]["expert_aucs"]
        labels, values, colors = [], [], []

        color_map = {
            "full_enrichment": "#4C72B0",
            "full_pathfinder": "#55A868",
            "pca":             "#C44E52",
            "coherence":       "#8172B2",
        }
        for name, val in sorted(expert_aucs.items()):
            if np.isnan(val):
                continue
            src_tag = "sed" if "sediment" in name else "rock"
            short = name.split("_", 2)[-1] if name.count("_") >= 2 else name
            expert_type = (
                "full_enrichment" if "full_enrichment" in name else
                "full_pathfinder" if "full_pathfinder" in name else
                "pca" if name.endswith("_pca") else
                "coherence" if name.endswith("_coherence") else "other"
            )
            labels.append(f"{src_tag}/{short}")
            values.append(val)
            colors.append(color_map.get(expert_type, "#cccccc"))

        labels.append("G(x) original")
        values.append(r["orig"]["auc"])
        colors.append("#FFA500")

        labels.append("G(x) all features")
        values.append(r["full"]["auc"])
        colors.append("#2ca02c")

        ax.barh(labels, values, color=colors, alpha=0.85)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xlim(0.2, 1.0)
        ax.set_title(
            f"{target}  orig={r['orig']['auc']:.3f} → full={r['full']['auc']:.3f}"
            f"  (test={r['n_test_pos']}pos/{r['n_test_neg']}neg)"
        )
        ax.set_xlabel("AUC")

    plt.tight_layout()
    plot_path = REPORTS / "phase3_all_features.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    logger.info("Saved plot: %s", plot_path)

    # ── write markdown report ─────────────────────────────────────────────────
    lines = [
        "# Phase 3 — All-Features vs. Original Expert Comparison\n",
        "## New expert types (full-feature config)",
        "",
        "| Expert | Features used | vs. original |",
        "|--------|--------------|--------------|",
        "| `FullEnrichmentExpert` | median + p90 + cv + frac_above + contrast (5/10 km) — 10 features | +median, +cv |",
        "| `FullPathfinderExpert` | same 10 stats × n_pathfinders + ratio features | +median, +cv per element |",
        "| `PCAExpert` | PC1–5 median at 5 km and 10 km — 10 features | entirely new |",
        "| `CoherenceExpert` | spatial_coherence_10km (negated) — 1 feature | entirely new |",
        "",
        "## AUC Comparison",
        "",
        "| Target | Orig AUC | Full AUC | Δ AUC | P@20 orig | P@20 full |",
        "|--------|----------|----------|-------|-----------|-----------|",
    ]
    for target, r in all_results.items():
        delta = r["full"]["auc"] - r["orig"]["auc"]
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| {target} | {r['orig']['auc']:.3f} | **{r['full']['auc']:.3f}** "
            f"| {sign}{delta:.3f} | {r['orig']['prec20']:.3f} | {r['full']['prec20']:.3f} |"
        )

    lines += ["", "## Per-expert AUC (full-feature config)", ""]
    for target, r in all_results.items():
        lines.append(f"### {target}")
        lines.append("| Expert | Source | AUC |")
        lines.append("|--------|--------|-----|")
        for name, val in sorted(r["full"]["expert_aucs"].items()):
            src = "sediment" if "sediment" in name else "rockchips"
            short = name.replace(f"{target.lower()}_{src}_", "").replace(f"{src}_", "")
            lines.append(f"| {short} | {src} | {val:.3f} |")
        lines.append("")

    lines += [
        "## Expert design notes",
        "",
        "### FullEnrichmentExpert vs TargetEnrichmentExpert",
        "Adds `{target}_{5/10}km_median` (absolute concentration level in log-space)"
        " and `{target}_{5/10}km_cv` (within-neighbourhood variability).",
        "CV is a hotspot indicator: high variability at small scale with low variability"
        " at larger scale suggests a concentrated source.",
        "",
        "### FullPathfinderExpert vs TargetPathfinderExpert",
        "Same extension applied to all pathfinder elements: 4 stats → 10 stats per element.",
        "",
        "### PCAExpert",
        "The global PCA is fitted on all assay samples in the source CSV (log1p + StandardScaler).",
        "PC scores at 5 km and 10 km capture neighbourhood geochemical assemblage.",
        "z-score normalisation handles sign ambiguity per PC component.",
        "",
        "### CoherenceExpert",
        "Measures mean pairwise distance (in degrees) of the top-5 primary-element samples",
        "within 10 km. **Negated** in scoring because lower distance = more spatially clustered",
        "= stronger anomaly signal.  Note: primary element is always `elements[0]` (Cu in the",
        "default element list), so this expert is most meaningful for Cu targeting.",
    ]

    report_path = REPORTS / "phase3_all_features.md"
    report_path.write_text("\n".join(lines) + "\n")
    logger.info("Saved report: %s", report_path)


if __name__ == "__main__":
    main()
