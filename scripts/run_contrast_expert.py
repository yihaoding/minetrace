"""run_contrast_expert.py

Data-driven ContrastExpert:
  1. Compute per-feature AUC (with direction) on KB positives vs background.
  2. Select top-K anomaly features (AUC > 0.5) and top-K background features
     (AUC < 0.5).
  3. Score = sigmoid( mean_z(anomaly_feats) − mean_z(bg_feats) )
     → anomaly evidence pushes score up; background evidence pulls it down.

Comparison per target:
  Original : TargetEnrichmentExpert + TargetPathfinderExpert × 2 sources (4 experts)
  Contrast : ContrastExpert × 2 sources (2 experts, data-driven features)

Usage:
    cd /group/pmc050/yding/gad_reasoning
    python scripts/run_contrast_expert.py

Output:
    reports/contrast_expert.md
    reports/contrast_{target}_{source}.png   (feature AUC bar chart)
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
GEO_DIR   = ROOT / "datasets/geochemical/states/state_geochemical"
REPORTS   = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

SED_CSV  = GEO_DIR / "gswa_all_sediment.csv"
ROCK_CSV = GEO_DIR / "gswa_all_rockchips.csv"

# ── config ────────────────────────────────────────────────────────────────────
K_FEATURES    = 10    # top-K anomaly + top-K background features
AUC_MIN_ANOM  = 0.55  # feature must reach this AUC to count as anomaly evidence
AUC_MAX_BG    = 0.45  # feature must be this low to count as background evidence
NAN_THRESH    = 0.30  # drop feature if >30% values are NaN
N_TRAIN_BG    = 300
MIN_COVERAGE  = 3
N_POS_TEST_FRAC = 0.10
N_POS_TEST_MIN  = 30
SEED = 42

TARGETS: dict[str, dict] = {
    "Cu": {
        "sites_csv": SITES_DIR / "gswa_cu_sites.csv",
        "site_col":  "Cu_SITES",
        "confidence_map": {"Mine": 1.0, "Deposit": 1.0},
        "pathfinders": {"Au": 2.0, "Mo": 1.5, "Ag": 1.0, "Co": 0.8, "Bi": 0.7},
        "ratio_features": ["ratio_Cu_Mo", "ratio_Cu_Pb", "ratio_Co_Ni"],
    },
    "Au": {
        "sites_csv": SITES_DIR / "gswa_au_site.csv",
        "site_col":  "Au_SITES",
        "confidence_map": {"Deposit": 1.0, "Prospect": 0.5},
        "pathfinders": {"Ag": 1.5, "As": 1.0, "Bi": 0.8, "Mo": 0.6},
        "ratio_features": ["ratio_Au_Cu", "ratio_Au_As"],
    },
    "W": {
        "sites_csv": SITES_DIR / "gswa_w_site.csv",
        "site_col":  "W_SITES",
        "confidence_map": {"Mine": 1.0, "Deposit": 1.0, "Prospect": 0.5},
        "pathfinders": {"Sn": 1.5, "Mo": 1.2, "Bi": 1.0, "As": 0.8},
        "ratio_features": ["ratio_W_Sn", "ratio_W_Mo"],
    },
    "Ni": {
        "sites_csv": SITES_DIR / "gswa_ni_site.csv",
        "site_col":  "Ni_SITES",
        "confidence_map": {"Mine": 1.0, "Deposit": 1.0, "Prospect": 0.5},
        "pathfinders": {"Co": 2.0, "Cr": 1.5, "Cu": 0.8},
        "ratio_features": ["ratio_Ni_Co", "ratio_Ni_Cr"],
    },
}


# ── ContrastExpert ────────────────────────────────────────────────────────────

class ContrastExpert:
    """Data-driven anomaly scoring by feature contrast.

    Uses two pre-selected feature sets:
      anomaly_feats  — high value → anomalous  (positive z-score contribution)
      bg_feats       — high value → background  (negative z-score contribution)

    score = sigmoid( mean_z(anomaly) − mean_z(background) )
    """

    def __init__(
        self,
        name: str,
        source_name: str,
        anomaly_feats: list[str],
        bg_feats: list[str],
    ) -> None:
        self.name = name
        self.required_sources: list[str] = [source_name]
        self._source_name = source_name
        self._anomaly_feats = anomaly_feats
        self._bg_feats = bg_feats
        self._bg_mean: dict[str, float] = {}
        self._bg_std:  dict[str, float] = {}
        self._fitted = False

    def fit_kb(self, background_samples: list) -> None:
        src = DataSourceRegistry.get(self._source_name)
        all_feats = self._anomaly_feats + self._bg_feats
        values: dict[str, list[float]] = {f: [] for f in all_feats}
        for s in background_samples:
            fv = src.get_features(s.id)
            if fv is None:
                continue
            for f in all_feats:
                v = fv.get(f, np.nan)
                if np.isfinite(v):
                    values[f].append(v)
        for f in all_feats:
            arr = np.array(values[f])
            self._bg_mean[f] = float(arr.mean()) if len(arr) > 0 else 0.0
            self._bg_std[f]  = float(arr.std())  if len(arr) > 1 else 1.0
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

        z_anom = [
            (fv[f] - self._bg_mean[f]) / self._bg_std[f]
            for f in self._anomaly_feats
            if np.isfinite(fv.get(f, np.nan))
        ]
        z_bg = [
            (fv[f] - self._bg_mean[f]) / self._bg_std[f]
            for f in self._bg_feats
            if np.isfinite(fv.get(f, np.nan))
        ]

        if not z_anom and not z_bg:
            return None

        signal = 0.0
        if z_anom:
            signal += float(np.mean(z_anom))
        if z_bg:
            signal -= float(np.mean(z_bg))   # background evidence subtracts

        return float(1.0 / (1.0 + np.exp(-signal)))


# ── feature selection ─────────────────────────────────────────────────────────

def select_contrast_features(
    positive_samples: list[GeochemSample],
    background_samples: list[GeochemSample],
    source: SedimentSource,
    k: int = K_FEATURES,
    auc_min_anom: float = AUC_MIN_ANOM,
    auc_max_bg: float = AUC_MAX_BG,
) -> tuple[list[str], list[str], pd.Series]:
    """
    Compute per-feature AUC (with direction) and split into anomaly / background sets.

    Returns
    -------
    anomaly_feats : top-k features with AUC > auc_min_anom
    bg_feats      : top-k features with AUC < auc_max_bg
    auc_series    : full per-feature AUC Series (sorted descending)
    """
    rows, labels = [], []
    for s in positive_samples:
        fv = source.get_features(s.id)
        if fv:
            rows.append(fv); labels.append(1)
    for s in background_samples:
        fv = source.get_features(s.id)
        if fv:
            rows.append(fv); labels.append(0)

    X = pd.DataFrame(rows)
    y = np.array(labels, dtype=int)

    # drop coverage features and high-NaN columns
    drop = [c for c in X.columns if c.startswith("n_samples_")]
    X = X.drop(columns=drop)
    nan_frac = X.isnull().mean()
    X = X[nan_frac[nan_frac <= NAN_THRESH].index]
    X = X.fillna(X.median())

    aucs: dict[str, float] = {}
    for col in X.columns:
        try:
            aucs[col] = float(roc_auc_score(y, X[col]))
        except Exception:
            aucs[col] = 0.5

    auc_series = pd.Series(aucs).sort_values(ascending=False)

    # anomaly features: highest AUC (clearly above 0.5)
    anomaly_feats = list(auc_series[auc_series >= auc_min_anom].head(k).index)
    # background features: lowest AUC (clearly below 0.5)
    bg_feats = list(auc_series[auc_series <= auc_max_bg].tail(k).index)

    return anomaly_feats, bg_feats, auc_series


# ── data helpers ──────────────────────────────────────────────────────────────

def load_instances(
    sites_csv: Path, elem_col: str, confidence_map: dict[str, float]
) -> tuple[list[GeochemInstance], list[GeochemInstance]]:
    df = pd.read_csv(sites_csv)
    high_conf, weak = [], []
    for _, row in df.iterrows():
        val = str(row.get(elem_col, ""))
        conf = max((c for kw, c in confidence_map.items() if kw in val), default=0.0)
        if conf == 0.0:
            continue
        site_code = str(row.get("SITE_CODE", row.get("fid", "")))
        try:
            x, y = float(row["X"]), float(row["Y"])
        except (ValueError, KeyError):
            continue
        s = GeochemSample(site_code=site_code, x=x, y=y,
                          site_stage=str(row.get("SITE_STAGE", "")),
                          site_type=str(row.get("SITE_TYPE_", "")))
        inst = GeochemInstance(sample=s, confidence=conf, metadata={})
        (high_conf if conf >= 1.0 else weak).append(inst)
    return high_conf, weak


def build_test_set(
    high_conf: list[GeochemInstance],
    weak: list[GeochemInstance],
    sed_src: SedimentSource,
    seed: int,
) -> tuple[list[GeochemSample], list[int], list[GeochemInstance]]:
    rng = np.random.default_rng(seed)
    n_test = min(max(N_POS_TEST_MIN, int(len(high_conf) * N_POS_TEST_FRAC)), len(high_conf))
    neg: list[GeochemSample] = []
    attempts = 0
    while len(neg) < n_test and attempts < n_test * 300:
        attempts += 1
        x = float(rng.uniform(WA_X_MIN, WA_X_MAX))
        y = float(rng.uniform(WA_Y_MIN, WA_Y_MAX))
        if sed_src.compute_for_point(x, y).get("n_samples_10km", 0) < MIN_COVERAGE:
            continue
        neg.append(GeochemSample(site_code=f"NEG_{len(neg):04d}", x=x, y=y))
    sed_src.register_samples(neg)
    idx = rng.choice(len(high_conf), size=n_test, replace=False)
    test_ids = {high_conf[i].sample.id for i in idx}
    test_pos = [high_conf[i].sample for i in idx]
    sed_src.register_samples(test_pos)
    train = [inst for inst in high_conf if inst.sample.id not in test_ids] + weak
    return test_pos + neg, [1] * len(test_pos) + [0] * len(neg), train


def generate_train_bg(sed_src: SedimentSource, n: int, seed: int) -> list[GeochemSample]:
    rng = np.random.default_rng(seed)
    samples: list[GeochemSample] = []
    attempts = 0
    while len(samples) < n and attempts < n * 300:
        attempts += 1
        x = float(rng.uniform(WA_X_MIN, WA_X_MAX))
        y = float(rng.uniform(WA_Y_MIN, WA_Y_MAX))
        if sed_src.compute_for_point(x, y).get("n_samples_10km", 0) < MIN_COVERAGE:
            continue
        samples.append(GeochemSample(site_code=f"TRBG_{len(samples):04d}", x=x, y=y))
    sed_src.register_samples(samples)
    return samples


# ── visualisation ─────────────────────────────────────────────────────────────

def plot_feature_aucs(
    auc_series: pd.Series,
    anomaly_feats: list[str],
    bg_feats: list[str],
    title: str,
    save_path: Path,
    n_show: int = 30,
) -> None:
    """Bar chart of top/bottom features by AUC, coloured by role."""
    top = auc_series.head(n_show // 2)
    bot = auc_series.tail(n_show // 2).sort_values()
    combined = pd.concat([top, bot]).sort_values(ascending=True)

    colors = []
    for feat in combined.index:
        if feat in anomaly_feats:
            colors.append("#2ca02c")   # green = anomaly evidence
        elif feat in bg_feats:
            colors.append("#d62728")   # red = background evidence
        else:
            colors.append("#aec7e8")   # grey = not selected

    fig, ax = plt.subplots(figsize=(9, max(6, len(combined) * 0.28)))
    bars = ax.barh(combined.index, combined.values - 0.5, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("AUC − 0.5  (positive = anomaly evidence, negative = background evidence)")
    ax.set_title(title, fontsize=11)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", alpha=0.85, label=f"Selected anomaly features (top-{K_FEATURES})"),
        Patch(facecolor="#d62728", alpha=0.85, label=f"Selected background features (top-{K_FEATURES})"),
        Patch(facecolor="#aec7e8", alpha=0.85, label="Not selected"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    logger.info("Saved: %s", save_path)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("Loading sediment source …")
    sed_src = SedimentSource(SED_CSV, name="sediment")
    logger.info("Loading rockchips source …")
    rock_src = SedimentSource(ROCK_CSV, name="rockchips")

    DataSourceRegistry.register(sed_src)
    DataSourceRegistry.register(rock_src)

    logger.info("Generating training background (%d points) …", N_TRAIN_BG)
    train_bg = generate_train_bg(sed_src, N_TRAIN_BG, seed=99)
    rock_src.register_samples(train_bg)

    all_results: dict[str, dict] = {}

    for target, cfg in TARGETS.items():
        logger.info("=" * 60)
        logger.info("TARGET: %s", target)

        high_conf, weak = load_instances(cfg["sites_csv"], cfg["site_col"], cfg["confidence_map"])
        test_samples, test_labels, train_instances = build_test_set(
            high_conf, weak, sed_src, seed=SEED)
        all_target_samples = test_samples + [inst.sample for inst in train_instances]
        rock_src.register_samples(all_target_samples)

        train_pos = [inst.sample for inst in train_instances]
        all_train  = train_pos + train_bg

        # ── feature selection per source ──────────────────────────────────
        feat_info: dict[str, dict] = {}
        for src_name, src in [("sediment", sed_src), ("rockchips", rock_src)]:
            logger.info("  Feature selection: %s / %s", target, src_name)
            a_feats, b_feats, auc_s = select_contrast_features(
                train_pos, train_bg, src)
            logger.info("    anomaly features (%d): %s", len(a_feats),
                        ", ".join(f"{f}({auc_s[f]:.3f})" for f in a_feats))
            logger.info("    background features (%d): %s", len(b_feats),
                        ", ".join(f"{f}({auc_s[f]:.3f})" for f in b_feats))
            feat_info[src_name] = {
                "anomaly_feats": a_feats, "bg_feats": b_feats, "auc_series": auc_s}
            plot_feature_aucs(
                auc_s, a_feats, b_feats,
                title=f"{target} / {src_name}  (green=anomaly, red=background evidence)",
                save_path=REPORTS / f"contrast_{target}_{src_name}.png",
            )

        # ── ORIGINAL pipeline ─────────────────────────────────────────────
        ExpertRegistry.clear()
        for src_name in ("sediment", "rockchips"):
            ExpertRegistry.register(TargetEnrichmentExpert(target, src_name))
            ExpertRegistry.register(TargetPathfinderExpert(
                target, src_name, cfg["pathfinders"],
                ratio_features=cfg.get("ratio_features", [])))
        orig_pipeline = L1Pipeline()
        orig_pipeline.fit(all_train, train_instances,
                          background_fn=ExcludeInstancesBackground())
        orig_results = orig_pipeline.score_all(test_samples)
        orig_auc  = compute_auc(orig_results, test_labels)
        orig_p20  = top_k_precision(orig_results, test_labels,
                                    min(20, len(test_labels) // 2))
        logger.info("  [original]  AUC=%.3f  P@20=%.3f", orig_auc, orig_p20)

        # ── CONTRAST pipeline ─────────────────────────────────────────────
        ExpertRegistry.clear()
        for src_name in ("sediment", "rockchips"):
            fi = feat_info[src_name]
            if not fi["anomaly_feats"] and not fi["bg_feats"]:
                logger.warning("  No discriminative features for %s/%s — skipping",
                               target, src_name)
                continue
            ExpertRegistry.register(ContrastExpert(
                name=f"{target.lower()}_{src_name}_contrast",
                source_name=src_name,
                anomaly_feats=fi["anomaly_feats"],
                bg_feats=fi["bg_feats"],
            ))
        cont_pipeline = L1Pipeline()
        cont_pipeline.fit(all_train, train_instances,
                          background_fn=ExcludeInstancesBackground())
        cont_results = cont_pipeline.score_all(test_samples)
        cont_auc = compute_auc(cont_results, test_labels)
        cont_p20 = top_k_precision(cont_results, test_labels,
                                   min(20, len(test_labels) // 2))
        cont_expert_aucs = {
            name: expert_auc(cont_results, test_labels, name)
            for name in cont_pipeline.active_expert_names
        }
        logger.info("  [contrast]  AUC=%.3f  P@20=%.3f", cont_auc, cont_p20)

        all_results[target] = {
            "orig_auc": orig_auc, "orig_p20": orig_p20,
            "cont_auc": cont_auc, "cont_p20": cont_p20,
            "cont_expert_aucs": cont_expert_aucs,
            "feat_info": feat_info,
            "n_pos": sum(test_labels),
            "n_neg": len(test_labels) - sum(test_labels),
        }

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n========== CONTRAST EXPERT vs ORIGINAL ==========")
    print(f"{'Target':<6}  {'Orig AUC':>9}  {'Cont AUC':>9}  {'Δ AUC':>7}  "
          f"{'P@20 orig':>9}  {'P@20 cont':>9}")
    print("-" * 60)
    for target, r in all_results.items():
        delta = r["cont_auc"] - r["orig_auc"]
        sign = "+" if delta >= 0 else ""
        print(f"{target:<6}  {r['orig_auc']:>9.3f}  {r['cont_auc']:>9.3f}"
              f"  {sign}{delta:>6.3f}  {r['orig_p20']:>9.3f}  {r['cont_p20']:>9.3f}")
    print("=" * 60)

    # ── write report ──────────────────────────────────────────────────────────
    lines = [
        "# ContrastExpert — Data-Driven Feature Selection\n",
        "## Scoring formula",
        "```",
        "anomaly_signal    = mean{ (feat - bg_mean) / bg_std  for feat in anomaly_feats }",
        "background_signal = mean{ (feat - bg_mean) / bg_std  for feat in bg_feats }",
        "score = sigmoid( anomaly_signal − background_signal )",
        "```",
        "",
        f"Selection: top-{K_FEATURES} features with AUC ≥ {AUC_MIN_ANOM} (anomaly) "
        f"and AUC ≤ {AUC_MAX_BG} (background), computed on KB positives vs random background.",
        "",
        "## AUC Comparison",
        "",
        "| Target | Orig AUC | Contrast AUC | Δ AUC | P@20 orig | P@20 cont |",
        "|--------|----------|--------------|-------|-----------|-----------|",
    ]
    for target, r in all_results.items():
        delta = r["cont_auc"] - r["orig_auc"]
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| {target} | {r['orig_auc']:.3f} | **{r['cont_auc']:.3f}** "
            f"| {sign}{delta:.3f} | {r['orig_p20']:.3f} | {r['cont_p20']:.3f} |"
        )

    lines += ["", "## Selected Features per Target / Source", ""]
    for target, r in all_results.items():
        lines.append(f"### {target}")
        for src_name, fi in r["feat_info"].items():
            auc_s = fi["auc_series"]
            lines.append(f"#### {src_name}")
            lines.append("")
            lines.append("**Anomaly features** (high value → anomalous):")
            lines.append("")
            lines.append("| Feature | AUC |")
            lines.append("|---------|-----|")
            for f in fi["anomaly_feats"]:
                lines.append(f"| `{f}` | {auc_s.get(f, float('nan')):.3f} |")
            lines.append("")
            lines.append("**Background features** (high value → background-like):")
            lines.append("")
            lines.append("| Feature | AUC |")
            lines.append("|---------|-----|")
            for f in fi["bg_feats"]:
                lines.append(f"| `{f}` | {auc_s.get(f, float('nan')):.3f} |")
            lines.append("")

    report_path = REPORTS / "contrast_expert.md"
    report_path.write_text("\n".join(lines) + "\n")
    logger.info("Saved report: %s", report_path)


if __name__ == "__main__":
    main()
