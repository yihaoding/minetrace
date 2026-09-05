"""Phase 2 v2 evaluation — sediment-based geochemical features.

Test set: 100 Cu Mine/Deposit (positive) + 100 random WA coordinates (negative).
Train/KB: remaining Cu Mine/Deposit sites.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_source import DataSourceRegistry
from core.expert import ExpertRegistry
from core.pipeline import L1Pipeline
from domains.geochem.background import ExcludeInstancesBackground
from domains.geochem.evaluation import build_test_set, compute_auc, expert_auc, top_k_precision
from domains.geochem.experts.cu_enrichment import CuEnrichmentExpert
from domains.geochem.experts.pathfinder import PathfinderExpert
from domains.geochem.evaluation import WA_X_MIN, WA_X_MAX, WA_Y_MIN, WA_Y_MAX
from domains.geochem.samples import GeochemSample
from domains.geochem.sources.sediment_source import SedimentSource
from domains.geochem.sources.site_catalog import SiteCatalogSource

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _generate_random_background(
    sed_src: SedimentSource,
    n: int,
    seed: int,
    min_coverage: int = 3,
) -> list[GeochemSample]:
    """Generate random WA coordinates as background samples for fit_kb.

    Uses a different seed from the test negatives to ensure independence.
    """
    rng = np.random.default_rng(seed)
    samples: list[GeochemSample] = []
    attempts = 0
    while len(samples) < n and attempts < n * 200:
        attempts += 1
        x = float(rng.uniform(WA_X_MIN, WA_X_MAX))
        y = float(rng.uniform(WA_Y_MIN, WA_Y_MAX))
        n_cover = sed_src.compute_for_point(x, y).get("n_samples_10km", 0)
        if n_cover < min_coverage:
            continue
        sid = f"TRAIN_BG_{len(samples):04d}"
        samples.append(GeochemSample(site_code=sid, x=x, y=y))
    if len(samples) < n:
        logger.warning("Only found %d/%d training background points", len(samples), n)
    return samples

SEDIMENT_CSV = Path(__file__).parent.parent / "datasets/geochemical/states/state_geochemical/gswa_all_sediment.csv"
SITES_CSV = Path(__file__).parent.parent / "datasets/geochemical/states/sites/gswa_cu_sites.csv"
REPORTS = Path(__file__).parent.parent / "reports"
REPORTS.mkdir(exist_ok=True)


def main() -> None:
    # ── load site catalog ─────────────────────────────────────────────────
    catalog = SiteCatalogSource(SITES_CSV)
    all_instances = catalog.load_cu_instances()
    logger.info("Catalog: %d Cu Mine/Deposit instances", len(all_instances))

    # ── init sediment source (builds KD-tree + PCA) ───────────────────────
    sed_src = SedimentSource(SEDIMENT_CSV)

    # ── register positive sites ───────────────────────────────────────────
    sed_src.register_samples([inst.sample for inst in all_instances])

    # ── build test set (100 pos + 100 random neg, seed=42) ────────────────
    test_samples, test_labels, train_instances = build_test_set(
        all_instances, sed_src, n_pos=100, n_neg=100, seed=42
    )

    # ── generate training background: random WA coordinates (seed≠42) ─────
    # Separate from test negatives; same distribution (random geographic coords).
    train_bg_samples = _generate_random_background(
        sed_src, n=500, seed=99, min_coverage=3
    )
    sed_src.register_samples(train_bg_samples)
    logger.info("Training background: %d random points", len(train_bg_samples))

    # pipeline sees: train positives + train background
    train_positive_samples = [inst.sample for inst in train_instances]
    all_train_samples = train_positive_samples + train_bg_samples

    # ── register sources and experts ──────────────────────────────────────
    DataSourceRegistry.register(sed_src)
    ExpertRegistry.register(CuEnrichmentExpert())
    ExpertRegistry.register(PathfinderExpert())

    # ── fit pipeline ──────────────────────────────────────────────────────
    # background_fn excludes instances → only random BG points remain
    bg_fn = ExcludeInstancesBackground()
    pipeline = L1Pipeline()
    pipeline.fit(all_train_samples, train_instances, background_fn=bg_fn)

    # ── score test set ────────────────────────────────────────────────────
    test_results = pipeline.score_all(test_samples)

    # ── metrics ──────────────────────────────────────────────────────────
    auc_combined = compute_auc(test_results, test_labels)
    expert_aucs = {
        name: expert_auc(test_results, test_labels, name)
        for name in pipeline.active_expert_names
    }
    prec20 = top_k_precision(test_results, test_labels, 20)
    prec50 = top_k_precision(test_results, test_labels, 50)

    base_rate = sum(test_labels) / len(test_labels)

    # ── print results ─────────────────────────────────────────────────────
    print("\n========== PHASE 2 v2 RESULTS (sediment features) ==========")
    print(f"Test set: {len(test_samples)} points | {sum(test_labels)} pos / {len(test_labels)-sum(test_labels)} neg")
    print(f"Train instances (KB): {len(train_instances)}")
    print()
    print("AUC:")
    print(f"  Random baseline:      0.500")
    for name, auc in expert_aucs.items():
        print(f"  Expert [{name}]: {auc:.3f}")
    print(f"  G(x) combined:        {auc_combined:.3f}")
    print()
    print(f"Top-20 precision: {prec20:.3f}  (base rate: {base_rate:.3f})")
    print(f"Top-50 precision: {prec50:.3f}")
    print("=============================================================\n")

    # ── plots ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    pos_scores = [r.g_score for r, l in zip(test_results, test_labels) if l == 1]
    neg_scores = [r.g_score for r, l in zip(test_results, test_labels) if l == 0]
    axes[0].hist(neg_scores, bins=30, alpha=0.6, label=f"Random BG (n={len(neg_scores)})", color="steelblue")
    axes[0].hist(pos_scores, bins=30, alpha=0.7, label=f"Cu Mine/Deposit (n={len(pos_scores)})", color="firebrick")
    axes[0].set_xlabel("G(x) score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("G(x) distribution — sediment features")
    axes[0].legend()

    aucs_plot = {**expert_aucs, "G(x) combined": auc_combined, "Random": 0.5}
    axes[1].barh(list(aucs_plot.keys()), list(aucs_plot.values()),
                 color=["#4C72B0", "#DD8452", "#55A868", "#8172B2", "gray"])
    axes[1].axvline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel("AUC (100 pos + 100 random neg)")
    axes[1].set_title("Expert AUC comparison")
    axes[1].set_xlim(0.3, 1.0)

    plt.tight_layout()
    plot_path = REPORTS / "phase_2v2_score_dist.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    logger.info("Saved plot: %s", plot_path)

    # ── report ────────────────────────────────────────────────────────────
    lines = [
        "# Phase 2 v2 Summary — Sediment Geochemical Features\n",
        "## Test Set Design",
        "- 100 positive: randomly selected Cu Mine/Deposit sites",
        "- 100 negative: randomly generated WA coordinates (≥5 sediment samples within 10 km)",
        f"- Train / KB instances: {len(train_instances)} Cu Mine/Deposit sites",
        "",
        "## Feature Design",
        "- Elements: Cu, Au, Mo, Ag, Pb, Zn, Co, Bi, Ni (9 elements)",
        "- Scales: 10 km (local), 25 km (district), 50 km (regional background)",
        "- Per element per scale: median, p90, CV, frac_above_threshold (log-space)",
        "- Multi-scale contrast: log ratio 10 km / 50 km (enrichment vs background)",
        "- Element ratios: Cu/Mo, Cu/Pb, Au/Cu, Zn/Pb, Co/Ni",
        "- Spatial coherence of top Cu samples within 10 km",
        "- Global PCA top-5 components, median at 10 km and 25 km",
        f"- Total dimensions: ~{len(sed_src.feature_names)}",
        "",
        "## AUC Results",
        "",
        "| Method | AUC |",
        "|--------|-----|",
        "| Random baseline | 0.500 |",
    ]
    for name, auc in expert_aucs.items():
        lines.append(f"| Expert: {name} | {auc:.3f} |")
    lines.append(f"| **G(x) combined** | **{auc_combined:.3f}** |")
    lines += [
        "",
        f"## Top-K Precision",
        f"- Top-20: {prec20:.3f} (base rate: {base_rate:.3f})",
        f"- Top-50: {prec50:.3f}",
    ]

    report_path = REPORTS / "phase_2v2_summary.md"
    report_path.write_text("\n".join(lines) + "\n")
    logger.info("Saved report: %s", report_path)


if __name__ == "__main__":
    main()
