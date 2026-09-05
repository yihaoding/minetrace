"""Phase 3: Multi-target, multi-source geochemical prospectivity evaluation.

Targets   : Cu, Au, W, Ni
Sources   : sediment (stream sediment) + rockchips (rock chip assays)
Test size : ~40 positive + 40 random negative per target
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

# ── target configuration ──────────────────────────────────────────────────────
# pathfinders: geochemically relevant co-enrichment elements per commodity system
TARGETS: dict[str, dict] = {
    "Cu": {
        "sites_csv": SITES_DIR / "gswa_cu_sites.csv",
        "site_col": "Cu_SITES",
        # confidence_map: keywords in *_SITES value → confidence
        # high-conf (1.0) used for both test + train; low-conf only for train KB
        "confidence_map": {"Mine": 1.0, "Deposit": 1.0},
        "pathfinders": {"Au": 2.0, "Mo": 1.5, "Ag": 1.0, "Co": 0.8, "Bi": 0.7},
        "ratio_features": ["ratio_Cu_Mo", "ratio_Cu_Pb", "ratio_Co_Ni"],
    },
    "Au": {
        "sites_csv": SITES_DIR / "gswa_au_site.csv",
        "site_col": "Au_SITES",
        # No Au Mine exists in this dataset — only Deposit + Prospect.
        # Deposit → test + train (high-conf); Prospect → train only (weak positive)
        "confidence_map": {"Deposit": 1.0, "Prospect": 0.5},
        # WA orogenic gold: Ag is the strongest sediment co-pathfinder;
        # As/Sb are useful in rockchips but weak in stream sediment here.
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

N_POS_TEST_FRAC = 0.10   # 10% of high-conf instances
N_POS_TEST_MIN = 30      # floor
N_TRAIN_BG = 300
MIN_SED_COVERAGE = 3


# ── helpers ───────────────────────────────────────────────────────────────────

def load_instances(
    sites_csv: Path,
    elem_col: str,
    confidence_map: dict[str, float] | None = None,
) -> tuple[list[GeochemInstance], list[GeochemInstance]]:
    """Return (high_conf_instances, weak_instances).

    high_conf: confidence == 1.0 → eligible for test set and train KB
    weak: confidence < 1.0 → train KB only (e.g. Prospects)
    """
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
    """Build test set: negatives first (random WA), then positives from high-conf.

    Test size = max(N_POS_TEST_MIN, floor(|high_conf| * N_POS_TEST_FRAC)).
    Equal number of random negatives.
    """
    rng = np.random.default_rng(seed)
    n_test = max(N_POS_TEST_MIN, int(len(high_conf) * N_POS_TEST_FRAC))
    n_test = min(n_test, len(high_conf))

    # ── 1. generate random negatives first ───────────────────────────────────
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

    # ── 2. select positive test samples ──────────────────────────────────────
    test_idx = rng.choice(len(high_conf), size=n_test, replace=False)
    test_pos_ids = {high_conf[i].sample.id for i in test_idx}
    test_pos = [high_conf[i].sample for i in test_idx]
    sed_src.register_samples(test_pos)

    # high-conf not in test + all weak instances → train KB
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


# ── explanation helpers ────────────────────────────────────────────────────────

def _load_site_titles(sites_csv: Path) -> dict[str, str]:
    df = pd.read_csv(sites_csv)
    result = {}
    for _, row in df.iterrows():
        code = str(row.get("SITE_CODE", row.get("fid", "")))
        title = str(row.get("SITE_TITLE", row.get("SHORT_NAME", code)))
        result[code] = title
    return result


def _feat_label(feat: str) -> str:
    """Convert internal feature name to a readable geochemical description."""
    if feat.startswith("ratio_"):
        parts = feat.split("_")
        return f"{parts[1]}/{parts[2]} log-ratio (10 km)"
    if feat.startswith("PC"):
        parts = feat.split("_")
        return f"PCA {parts[0]} score ({parts[1]})"
    if feat.startswith("n_samples_"):
        return f"Sample coverage ({feat.split('_')[2]})"
    if "_contrast_" in feat:
        elem, scales = feat.split("_contrast_")
        s1, s2 = scales.split("_")
        return f"{elem} local/BG contrast ({s1} km vs {s2} km)"
    if "km_" in feat:
        # e.g. Cu_10km_median, W_5km_frac_above
        idx = feat.rindex("km_")
        head = feat[:idx + 2]   # "Cu_10km"
        stat = feat[idx + 3:]   # "median"
        stat_map = {"median": "median", "p90": "90th pct", "cv": "CV", "frac_above": "frac>75th pct"}
        return f"{head.replace('_', ' ')} {stat_map.get(stat, stat)}"
    return feat


def compute_bg_stats(
    bg_samples: list[GeochemSample],
    sources: list[SedimentSource],
) -> dict[str, dict[str, tuple[float, float]]]:
    """Return {source_name: {feature: (mean, std)}} from background samples."""
    stats: dict[str, dict[str, tuple[float, float]]] = {}
    for src in sources:
        feat_vals: dict[str, list[float]] = {}
        for s in bg_samples:
            fv = src.get_features(s.id)
            if fv is None:
                continue
            for k, v in fv.items():
                if np.isfinite(v):
                    feat_vals.setdefault(k, []).append(v)
        stats[src.name] = {
            k: (float(np.mean(v)), max(float(np.std(v)), 1e-9))
            for k, v in feat_vals.items()
            if len(v) >= 10
        }
    return stats


def explain_site(
    sample: GeochemSample,
    sources: list[SedimentSource],
    bg_stats: dict[str, dict[str, tuple[float, float]]],
    n: int = 5,
) -> list[dict]:
    """Return top-n most anomalous features for a sample vs background."""
    reasons = []
    for src in sources:
        fv = src.get_features(sample.id)
        if fv is None:
            continue
        s_stats = bg_stats.get(src.name, {})
        for feat, val in fv.items():
            if not np.isfinite(val) or feat not in s_stats:
                continue
            # skip coverage features — not informative as reasons
            if feat.startswith("n_samples_"):
                continue
            mu, sigma = s_stats[feat]
            z = (val - mu) / sigma
            reasons.append({
                "feature": feat,
                "label": _feat_label(feat),
                "source": src.name,
                "z": z,
                "value": val,
                "bg_mean": mu,
            })
    # sort by z descending (strongest positive anomaly first)
    reasons.sort(key=lambda r: r["z"], reverse=True)
    return reasons[:n]


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load both assay sources once (same element set for both)
    logger.info("Loading sediment source …")
    sed_src = SedimentSource(SED_CSV, name="sediment")
    logger.info("Loading rockchips source …")
    rock_src = SedimentSource(ROCK_CSV, name="rockchips")

    DataSourceRegistry.register(sed_src)
    DataSourceRegistry.register(rock_src)

    # Shared training background (same points for all targets)
    logger.info("Generating shared training background …")
    train_bg = generate_train_bg(sed_src, N_TRAIN_BG, seed=99)
    rock_src.register_samples(train_bg)
    logger.info("Training background: %d points", len(train_bg))

    all_results: dict[str, dict] = {}

    for target, cfg in TARGETS.items():
        logger.info("=" * 60)
        logger.info("TARGET: %s", target)

        # Load positive instances (high-conf for test+train, weak for train only)
        high_conf, weak = load_instances(cfg["sites_csv"], cfg["site_col"], cfg.get("confidence_map"))
        logger.info("%s: %d high-conf, %d weak instances", target, len(high_conf), len(weak))

        # Build test set (10% of high-conf, min 30; equal negatives)
        test_samples, test_labels, train_instances = build_test_set(
            high_conf, weak, sed_src, seed=42,
        )

        # Register all test + train positives in rockchips source
        all_target_samples = test_samples + [inst.sample for inst in train_instances]
        rock_src.register_samples(all_target_samples)

        # Set up target-specific experts (fresh per target)
        ExpertRegistry.clear()
        for src_name in ("sediment", "rockchips"):
            ExpertRegistry.register(TargetEnrichmentExpert(target, src_name))
            ExpertRegistry.register(TargetPathfinderExpert(
                target, src_name, cfg["pathfinders"],
                ratio_features=cfg.get("ratio_features", []),
            ))

        # Fit pipeline
        train_positive_samples = [inst.sample for inst in train_instances]
        all_train = train_positive_samples + train_bg
        bg_fn = ExcludeInstancesBackground()
        pipeline = L1Pipeline()
        pipeline.fit(all_train, train_instances, background_fn=bg_fn)

        # Score test set
        test_results = pipeline.score_all(test_samples)

        # Metrics
        base_rate = sum(test_labels) / len(test_labels)
        auc = compute_auc(test_results, test_labels)
        expert_aucs = {
            name: expert_auc(test_results, test_labels, name)
            for name in pipeline.active_expert_names
        }
        prec20 = top_k_precision(test_results, test_labels, min(20, len(test_labels) // 2))
        prec40 = top_k_precision(test_results, test_labels, min(40, len(test_labels)))

        all_results[target] = {
            "n_instances": len(high_conf) + len(weak),
            "n_train_kb": len(train_instances),
            "n_test_pos": sum(test_labels),
            "n_test_neg": len(test_labels) - sum(test_labels),
            "auc": auc,
            "expert_aucs": expert_aucs,
            "prec20": prec20,
            "prec40": prec40,
            "base_rate": base_rate,
            "active_experts": list(pipeline.active_expert_names),
            # kept for explanation phase
            "test_samples": test_samples,
            "test_results": test_results,
            "test_labels": test_labels,
            "site_titles": _load_site_titles(cfg["sites_csv"]),
        }

        logger.info("  AUC=%.3f  prec@20=%.3f  prec@40=%.3f", auc, prec20, prec40)

    # ── print summary table ───────────────────────────────────────────────────
    print("\n========== PHASE 3 MULTI-TARGET RESULTS ==========")
    print(f"{'Target':<6} {'N_KB':>5} {'N_test':>6} {'AUC':>6} {'P@20':>6} {'P@40':>6} {'BaseRate':>8}")
    print("-" * 50)
    for target, r in all_results.items():
        print(
            f"{target:<6} {r['n_train_kb']:>5} {r['n_test_pos']+r['n_test_neg']:>6} "
            f"{r['auc']:>6.3f} {r['prec20']:>6.3f} {r['prec40']:>6.3f} {r['base_rate']:>8.3f}"
        )
    print("===================================================")
    print("\nPer-expert AUC breakdown:")
    for target, r in all_results.items():
        print(f"\n  {target}:")
        for exp_name, auc_val in sorted(r["expert_aucs"].items()):
            src = "sed" if "sediment" in exp_name else "rock"
            print(f"    [{src}] {exp_name}: {auc_val:.3f}")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (target, r) in zip(axes, all_results.items()):
        # group experts by source
        sed_aucs, rock_aucs = {}, {}
        for name, val in r["expert_aucs"].items():
            if np.isnan(val):
                continue
            if "sediment" in name:
                sed_aucs[name.replace(f"{target.lower()}_sediment_", "")] = val
            elif "rockchips" in name:
                rock_aucs[name.replace(f"{target.lower()}_rockchips_", "")] = val

        labels, values, colors = [], [], []
        for k, v in sed_aucs.items():
            labels.append(f"sed/{k}")
            values.append(v)
            colors.append("#4C72B0")
        for k, v in rock_aucs.items():
            labels.append(f"rock/{k}")
            values.append(v)
            colors.append("#DD8452")
        labels.append("G(x) combined")
        values.append(r["auc"])
        colors.append("#55A868")

        bars = ax.barh(labels, values, color=colors, alpha=0.85)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xlim(0.3, 1.0)
        ax.set_title(
            f"{target}  (AUC={r['auc']:.3f}, KB={r['n_train_kb']}, test={r['n_test_pos']}pos/{r['n_test_neg']}neg)"
        )
        ax.set_xlabel("AUC")

    plt.tight_layout()
    plot_path = REPORTS / "phase3_multitarget.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    logger.info("Saved plot: %s", plot_path)

    # ── write markdown report ─────────────────────────────────────────────────
    lines = [
        "# Phase 3 — Multi-Target Multi-Source Results\n",
        "## Configuration",
        f"- Targets: {', '.join(TARGETS.keys())}",
        "- Sources: sediment + rockchips",
        f"- Test size: max({N_POS_TEST_MIN}, 10% of high-conf) pos + equal neg per target",
        f"- Training background: {N_TRAIN_BG} random WA coordinates",
        "",
        "## Summary Table",
        "",
        "| Target | KB sites | Test (pos/neg) | AUC | P@20 | P@40 |",
        "|--------|----------|----------------|-----|------|------|",
    ]
    for target, r in all_results.items():
        lines.append(
            f"| {target} | {r['n_train_kb']} | {r['n_test_pos']}/{r['n_test_neg']} "
            f"| **{r['auc']:.3f}** | {r['prec20']:.3f} | {r['prec40']:.3f} |"
        )
    lines += ["", "## Per-Expert AUC", ""]
    for target, r in all_results.items():
        lines.append(f"### {target}")
        lines.append("| Expert | Source | AUC |")
        lines.append("|--------|--------|-----|")
        for name, val in sorted(r["expert_aucs"].items()):
            src = "sediment" if "sediment" in name else "rockchips"
            short = name.replace(f"{target.lower()}_{src}_", "")
            lines.append(f"| {short} | {src} | {val:.3f} |")
        lines.append("")

    # ── explanation phase ──────────────────────────────────────────────────────
    logger.info("Computing feature explanations …")
    bg_stats = compute_bg_stats(train_bg, [sed_src, rock_src])

    explanation_lines: list[str] = [
        "",
        "---",
        "",
        "## Methodology",
        "",
        "### Data sources",
        "- **Sediment** (`gswa_all_sediment.csv`, 157 k samples): stream-sediment geochemistry.",
        "  Wide spatial dispersion; good for regional anomaly detection.",
        "- **Rockchips** (`gswa_all_rockchips.csv`, 402 k samples): direct rock-chip assays.",
        "  High concentration values near mineralisation; strong near-field signal.",
        "",
        "### Feature engineering (per target point)",
        "For each point, samples are aggregated within **3 spatial radii** (5 km / 10 km / 50 km):",
        "",
        "| Feature group | Count | Description |",
        "|---------------|-------|-------------|",
        "| Per-element statistics | 14 elem × 3 scales × 4 stats = 168 | median, 90th-pct, CV, frac>75th-pct (log-space) |",
        "| Multi-scale contrast | 14 elem × 2 pairs = 28 | log(local) − log(regional BG): measures enrichment |",
        "| Element ratios | up to 12 pairs | log-ratio at 10 km (e.g. W/Sn, Co/Ni) |",
        "| Spatial coherence | 1 | mean pairwise distance of top-5 samples within 10 km |",
        "| Global PCA scores | 5 PC × 2 scales = 10 | 14-element PCA, median in neighbourhood |",
        "| Coverage | 3 | sample count at each scale |",
        f"| **Total** | **~250 dims** | per source per point |",
        "",
        "### Scoring pipeline",
        "- **2 experts per source**: `enrichment` (direct target-element anomaly) +",
        "  `pathfinder` (weighted co-element anomaly)",
        "- Each expert computes a **z-score** against the null distribution (300 random WA points)",
        "  then maps to [0,1] via sigmoid.",
        "- **G(x)** = sigmoid(mean z-score across all active experts)",
        "",
        "### Test set construction",
        "1. Generate random WA negatives first (require ≥3 sediment samples within 10 km)",
        "2. Randomly draw positives from high-confidence (Mine/Deposit) sites",
        "3. Test size = max(30, 10% of available high-conf sites) per target",
        "4. Remaining high-conf + all Prospect sites → training KB",
        "",
        "---",
        "",
        "## Top-5 Scored Positive Sites per Target (with Reasons)",
        "",
    ]

    for target, r in all_results.items():
        explanation_lines.append(f"### {target}")
        explanation_lines.append("")

        paired = [
            (res, samp, label)
            for res, samp, label in zip(r["test_results"], r["test_samples"], r["test_labels"])
            if label == 1
        ]
        top5 = sorted(paired, key=lambda t: t[0].g_score, reverse=True)[:5]

        for rank, (res, samp, _) in enumerate(top5, 1):
            title = r["site_titles"].get(samp.id, samp.id)
            explanation_lines.append(
                f"**#{rank} — {title}** (`{samp.id}`, {samp.x:.3f}°E {abs(samp.y):.3f}°S)  "
                f"G(x)={res.g_score:.3f}"
            )
            explanation_lines.append("")

            # expert scores breakdown
            exp_parts = [f"{n}: {v:.3f}" for n, v in sorted(res.expert_scores.items())]
            explanation_lines.append(f"Expert scores: {' | '.join(exp_parts)}")
            explanation_lines.append("")

            # top-5 feature reasons
            reasons = explain_site(samp, [sed_src, rock_src], bg_stats, n=5)
            explanation_lines.append("Top 5 anomalous features vs random background:")
            explanation_lines.append("")
            explanation_lines.append("| # | Feature | Source | z-score | Value | BG mean |")
            explanation_lines.append("|---|---------|--------|---------|-------|---------|")
            for i, rea in enumerate(reasons, 1):
                explanation_lines.append(
                    f"| {i} | {rea['label']} | {rea['source']} "
                    f"| **{rea['z']:+.1f}σ** | {rea['value']:.3f} | {rea['bg_mean']:.3f} |"
                )
            explanation_lines.append("")

        explanation_lines.append("")

    lines += explanation_lines

    report_path = REPORTS / "phase3_multitarget_summary.md"
    report_path.write_text("\n".join(lines) + "\n")
    logger.info("Saved report: %s", report_path)


if __name__ == "__main__":
    main()
