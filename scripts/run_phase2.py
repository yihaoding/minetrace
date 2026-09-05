"""Phase 2 evaluation script.

Runs the full L1 pipeline on gswa_cu_sites.csv, computes:
  - Per-expert AUC on spatial block holdout
  - Combined G(x) AUC
  - Top-20 / Top-50 precision
  - Extensibility demo (stub expert auto-skip)
  - Saves plots and reports/phase_2_summary.md
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
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_source import DataSourceRegistry
from core.expert import ExpertRegistry
from core.pipeline import L1Pipeline
from domains.geochem.background import ExcludeInstancesBackground
from domains.geochem.experts.commodity_affinity import CommodityAffinityExpert
from domains.geochem.experts.density import DensityExpert
from domains.geochem.experts.stage_anomaly import StageAnomalyExpert
from domains.geochem.sources.site_catalog import SiteCatalogSource

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CSV_PATH = Path(__file__).parent.parent / "datasets/geochemical/states/sites/gswa_cu_sites.csv"
REPORTS = Path(__file__).parent.parent / "reports"
REPORTS.mkdir(exist_ok=True)


def spatial_block_split(samples, instances, n_blocks: int = 10, test_frac: float = 0.2, seed: int = 42):
    """Split samples into train/test using spatial blocks.

    Divides the bounding box into n_blocks × n_blocks cells.
    Each cell is independently assigned to train (80%) or test (20%).
    Returns (train_samples, test_samples, train_instances, test_instances).
    """
    rng = np.random.default_rng(seed)
    xs = np.array([s.x for s in samples])
    ys = np.array([s.y for s in samples])
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    def cell_id(x, y):
        cx = int((x - x_min) / (x_max - x_min + 1e-9) * n_blocks)
        cy = int((y - y_min) / (y_max - y_min + 1e-9) * n_blocks)
        return min(cx, n_blocks - 1), min(cy, n_blocks - 1)

    all_cells = {(i, j) for i in range(n_blocks) for j in range(n_blocks)}
    n_test_cells = max(1, int(len(all_cells) * test_frac))
    test_cells = set(map(tuple, rng.choice(list(all_cells), size=n_test_cells, replace=False).tolist()))

    train_samples, test_samples = [], []
    for s in samples:
        if cell_id(s.x, s.y) in test_cells:
            test_samples.append(s)
        else:
            train_samples.append(s)

    instance_ids = {inst.sample.id for inst in instances}
    train_instances = [i for i in instances if i.sample.id in {s.id for s in train_samples}]
    test_instances = [i for i in instances if i.sample.id in {s.id for s in test_samples}]

    return train_samples, test_samples, train_instances, test_instances


def compute_auc(results, instance_ids):
    y_true = [1 if r.sample.id in instance_ids else 0 for r in results]
    y_score = [r.g_score for r in results]
    if sum(y_true) == 0 or sum(y_true) == len(y_true):
        return float("nan")
    return roc_auc_score(y_true, y_score)


def compute_expert_auc(results, expert_name, instance_ids):
    y_true, y_score = [], []
    for r in results:
        if expert_name in r.expert_scores:
            y_true.append(1 if r.sample.id in instance_ids else 0)
            y_score.append(r.expert_scores[expert_name])
    if len(y_true) == 0 or sum(y_true) == 0:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def top_k_precision(results, instance_ids, k: int):
    sorted_r = sorted(results, key=lambda r: r.g_score, reverse=True)
    top_k = sorted_r[:k]
    hits = sum(1 for r in top_k if r.sample.id in instance_ids)
    return hits / k


def main():
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    catalog = SiteCatalogSource(CSV_PATH)
    DataSourceRegistry.register(catalog)

    ExpertRegistry.register(DensityExpert())
    ExpertRegistry.register(CommodityAffinityExpert())
    ExpertRegistry.register(StageAnomalyExpert())

    all_samples = catalog.load_all_samples()
    all_instances = catalog.load_cu_instances()
    instance_ids = {i.sample.id for i in all_instances}
    base_rate = len(instance_ids) / len(all_samples)
    logger.info("Loaded %d samples, %d instances (base rate %.3f)", len(all_samples), len(instance_ids), base_rate)

    # ------------------------------------------------------------------
    # Spatial block split
    # ------------------------------------------------------------------
    train_s, test_s, train_i, test_i = spatial_block_split(all_samples, all_instances)
    logger.info("Train: %d samples (%d instances) | Test: %d samples (%d instances)",
                len(train_s), len(train_i), len(test_s), len(test_i))

    # ------------------------------------------------------------------
    # Fit pipeline on train, score test
    # ------------------------------------------------------------------
    bg_fn = ExcludeInstancesBackground()
    pipeline = L1Pipeline()
    pipeline.fit(train_s, train_i, background_fn=bg_fn)

    test_results = pipeline.score_all(test_s)
    train_results = pipeline.score_all(train_s)  # for distribution plot

    test_instance_ids = {i.sample.id for i in test_i}
    train_instance_ids = {i.sample.id for i in train_i}

    # ------------------------------------------------------------------
    # AUC table
    # ------------------------------------------------------------------
    results_table = {}
    n_total_test = len(test_results)
    n_pos_test = sum(1 for r in test_results if r.sample.id in test_instance_ids)
    base_rate_test = n_pos_test / n_total_test if n_total_test > 0 else 0

    results_table["random_baseline"] = 0.500
    for ename in pipeline.active_expert_names:
        results_table[f"expert_{ename}"] = compute_expert_auc(test_results, ename, test_instance_ids)
    results_table["G(x)_combined"] = compute_auc(test_results, test_instance_ids)

    top20_prec = top_k_precision(test_results, test_instance_ids, 20)
    top50_prec = top_k_precision(test_results, test_instance_ids, 50)

    # ------------------------------------------------------------------
    # Extensibility demo
    # ------------------------------------------------------------------
    class StubGeophysicsExpert:
        name = "stub_geophysics"
        required_sources = ["geophysics"]  # not registered
        def fit_kb(self, bg): pass
        def score(self, s): return 0.5

    ExpertRegistry.register(StubGeophysicsExpert())
    pipeline2 = L1Pipeline()
    pipeline2.fit(train_s[:50], train_i, background_fn=bg_fn)
    ext_active = pipeline2.active_expert_names
    ext_skipped = "stub_geophysics" not in ext_active

    # ------------------------------------------------------------------
    # Plot 1: G(x) score distribution (test set)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    pos_scores = [r.g_score for r in test_results if r.sample.id in test_instance_ids]
    neg_scores = [r.g_score for r in test_results if r.sample.id not in test_instance_ids]
    axes[0].hist(neg_scores, bins=40, alpha=0.6, label=f"Background (n={len(neg_scores)})", color="steelblue")
    axes[0].hist(pos_scores, bins=40, alpha=0.7, label=f"Cu Mine/Deposit (n={len(pos_scores)})", color="firebrick")
    axes[0].set_xlabel("G(x) score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("G(x) score distribution (test set)")
    axes[0].legend()

    # Plot 2: per-expert AUC bar chart
    expert_aucs = {k.replace("expert_", ""): v for k, v in results_table.items() if k.startswith("expert_")}
    expert_aucs["G(x) combined"] = results_table["G(x)_combined"]
    expert_aucs["Random"] = 0.5
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    axes[1].barh(list(expert_aucs.keys()), list(expert_aucs.values()), color=colors[:len(expert_aucs)])
    axes[1].axvline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel("AUC (spatial holdout)")
    axes[1].set_title("Per-expert AUC")
    axes[1].set_xlim(0.4, 1.0)

    plt.tight_layout()
    plot_path = REPORTS / "phase_2_score_dist.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    logger.info("Saved plot: %s", plot_path)

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------
    lines = [
        "# Phase 2 Summary\n",
        "## Dataset",
        f"- Total sites: {len(all_samples)}",
        f"- Positive instances (Cu Mine + Cu Deposit): {len(instance_ids)} (base rate {base_rate:.3f})",
        f"- Positive rule: `Cu_SITES` contains 'Mine' or 'Deposit' (DP-02 resolved)",
        "",
        "## Spatial Block Holdout Split",
        f"- 10×10 grid, 20% cells → test",
        f"- Train: {len(train_s)} samples ({len(train_i)} instances)",
        f"- Test: {len(test_s)} samples ({len(test_i)} instances, base rate {base_rate_test:.3f})",
        "",
        "## AUC Results (spatial holdout)",
        "",
        "| Method | AUC |",
        "|--------|-----|",
        f"| Random baseline | 0.500 |",
    ]
    for ename in pipeline.active_expert_names:
        auc = results_table.get(f"expert_{ename}", float("nan"))
        lines.append(f"| Expert: {ename} | {auc:.3f} |")
    lines.append(f"| **G(x) combined** | **{results_table['G(x)_combined']:.3f}** |")
    lines += [
        "",
        "## Top-K Precision (test set)",
        f"- Top-20 precision: {top20_prec:.3f} (random baseline: {base_rate_test:.3f})",
        f"- Top-50 precision: {top50_prec:.3f}",
        "",
        "## Active Experts",
        f"- Active: {pipeline.active_expert_names}",
        "",
        "## Extensibility Demo",
        "- Registered stub Expert requiring `geophysics` (not in DataSourceRegistry)",
        f"- Auto-skipped: {ext_skipped}",
        f"- Active experts with stub registered: {ext_active}",
        "- **Zero lines of core/pipeline.py changed**",
        "",
        "## Stage Score Mapping (DP-02 resolved)",
        "| SITE_STAGE | score |",
        "|------------|-------|",
        "| Operating | 5 |",
        "| Under Development | 4 |",
        "| Proposed | 3 |",
        "| Care and Maintenance | 2 |",
        "| Shut | 1 |",
        "| Undeveloped | 0 |",
        "",
        "## Files Added (Phase 2)",
        "- `domains/geochem/samples.py`",
        "- `domains/geochem/sources/site_catalog.py`",
        "- `domains/geochem/neighborhood.py`",
        "- `domains/geochem/background.py`",
        "- `domains/geochem/experts/density.py`",
        "- `domains/geochem/experts/commodity_affinity.py`",
        "- `domains/geochem/experts/stage_anomaly.py`",
        "- `core/pipeline.py`",
        "- `tests/domains/geochem/test_site_catalog.py`",
    ]

    report_path = REPORTS / "phase_2_summary.md"
    report_path.write_text("\n".join(lines) + "\n")
    logger.info("Saved report: %s", report_path)

    # Print summary to stdout
    print("\n========== PHASE 2 RESULTS ==========")
    print(f"Total sites: {len(all_samples)} | Positives: {len(instance_ids)} ({base_rate:.1%})")
    print(f"Test set: {len(test_s)} sites | {len(test_i)} positives")
    print()
    print("AUC (spatial holdout):")
    print(f"  Random baseline:      0.500")
    for ename in pipeline.active_expert_names:
        auc = results_table.get(f"expert_{ename}", float("nan"))
        print(f"  Expert [{ename}]: {auc:.3f}")
    print(f"  G(x) combined:        {results_table['G(x)_combined']:.3f}")
    print()
    print(f"Top-20 precision: {top20_prec:.3f}  (base rate: {base_rate_test:.3f})")
    print(f"Top-50 precision: {top50_prec:.3f}")
    print()
    print(f"Extensibility: stub_geophysics Expert auto-skipped = {ext_skipped}")
    print("=====================================\n")


if __name__ == "__main__":
    main()
