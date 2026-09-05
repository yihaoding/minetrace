"""run_feature_selection.py

Five feature selection methods for geochemical prospectivity.

For each target (Cu, Au, W, Ni) × source (sediment, rockchips):
  M1  Per-feature AUC              — univariate filter; max(AUC, 1-AUC)
  M2  Pearson corr-filter + AUC    — greedy remove |r|>0.9, then M1
  M3  Random Forest importance     — mean-decrease-impurity (balanced RF)
  M4  LASSO logistic regression    — |coef| from L1-penalised LR (CV-tuned C)
  M5  Greedy forward selection     — 5-fold CV AUC on M2 shortlist

Consensus: Borda-count rank across M1–M5 (lower = better).

Usage:
    cd /group/pmc050/yding/gad_reasoning
    python scripts/run_feature_selection.py

Output:
    reports/feature_selection.md
    reports/feature_selection_{target}_{source}.png  (rank heatmap)
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from domains.geochem.evaluation import WA_X_MIN, WA_X_MAX, WA_Y_MIN, WA_Y_MAX
from domains.geochem.samples import GeochemSample, GeochemInstance
from domains.geochem.sources.sediment_source import SedimentSource

warnings.filterwarnings("ignore")
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

# ── constants ─────────────────────────────────────────────────────────────────
N_TRAIN_BG = 300
MIN_SED_COVERAGE = 3
CORR_THRESHOLD = 0.90      # M2: drop feature if |r| > this with any kept feature
M5_CANDIDATES = 30         # M2 shortlist size fed into forward selection
M5_MAX_STEPS = 20          # max features selected by M5
M5_MIN_DELTA = 0.003       # stop M5 if CV AUC gain < this
N_TOP = 20                 # top-N features shown in report / heatmap
SEED = 42

# ── target config ─────────────────────────────────────────────────────────────
TARGETS: dict[str, dict] = {
    "Cu": {
        "sites_csv": SITES_DIR / "gswa_cu_sites.csv",
        "site_col":  "Cu_SITES",
        "confidence_map": {"Mine": 1.0, "Deposit": 1.0},
    },
    "Au": {
        "sites_csv": SITES_DIR / "gswa_au_site.csv",
        "site_col":  "Au_SITES",
        "confidence_map": {"Deposit": 1.0, "Prospect": 0.5},
    },
    "W": {
        "sites_csv": SITES_DIR / "gswa_w_site.csv",
        "site_col":  "W_SITES",
        "confidence_map": {"Mine": 1.0, "Deposit": 1.0, "Prospect": 0.5},
    },
    "Ni": {
        "sites_csv": SITES_DIR / "gswa_ni_site.csv",
        "site_col":  "Ni_SITES",
        "confidence_map": {"Mine": 1.0, "Deposit": 1.0, "Prospect": 0.5},
    },
}


# ── data helpers ──────────────────────────────────────────────────────────────

def load_instances(
    sites_csv: Path,
    elem_col: str,
    confidence_map: dict[str, float],
) -> list[GeochemInstance]:
    df = pd.read_csv(sites_csv)
    instances = []
    for _, row in df.iterrows():
        val = str(row.get(elem_col, ""))
        conf = max(
            (c for kw, c in confidence_map.items() if kw in val),
            default=0.0,
        )
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
        instances.append(GeochemInstance(sample=s, confidence=conf, metadata={}))
    return instances


def generate_background(sed_src: SedimentSource, n: int, seed: int) -> list[GeochemSample]:
    rng = np.random.default_rng(seed)
    samples: list[GeochemSample] = []
    attempts = 0
    while len(samples) < n and attempts < n * 300:
        attempts += 1
        x = float(rng.uniform(WA_X_MIN, WA_X_MAX))
        y = float(rng.uniform(WA_Y_MIN, WA_Y_MAX))
        if sed_src.compute_for_point(x, y).get("n_samples_10km", 0) < MIN_SED_COVERAGE:
            continue
        samples.append(GeochemSample(site_code=f"BG_{len(samples):04d}", x=x, y=y))
    sed_src.register_samples(samples)
    return samples


def collect_xy(
    positive_samples: list[GeochemSample],
    background_samples: list[GeochemSample],
    source: SedimentSource,
) -> tuple[pd.DataFrame, np.ndarray]:
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
    return X, y


def preprocess(X: pd.DataFrame, nan_thresh: float = 0.30) -> pd.DataFrame:
    """Drop coverage features and high-NaN columns; fill remaining NaN with median."""
    drop = [c for c in X.columns if c.startswith("n_samples_")]
    X = X.drop(columns=drop)
    nan_frac = X.isnull().mean()
    X = X[nan_frac[nan_frac <= nan_thresh].index]
    X = X.fillna(X.median())
    return X


# ── M1: per-feature AUC ───────────────────────────────────────────────────────

def m1_auc(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    aucs = {}
    for col in X.columns:
        try:
            a = roc_auc_score(y, X[col])
            aucs[col] = max(a, 1.0 - a)   # reflect below-0.5 features
        except Exception:
            aucs[col] = 0.5
    return pd.Series(aucs).sort_values(ascending=False)


# ── M2: correlation filter + AUC ─────────────────────────────────────────────

def m2_corr_auc(
    X: pd.DataFrame, y: np.ndarray, threshold: float = CORR_THRESHOLD
) -> pd.Series:
    auc_s = m1_auc(X, y)
    corr = X.corr().abs()
    kept: list[str] = []
    for feat in auc_s.index:
        if not kept or corr.loc[feat, kept].max() < threshold:
            kept.append(feat)
    return auc_s[kept].sort_values(ascending=False)


# ── M3: Random Forest importance ──────────────────────────────────────────────

def m3_rf(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=SEED, n_jobs=-1,
    )
    rf.fit(Xs, y)
    # report OOB AUC for reference (stored in oob_decision_function_)
    try:
        rf2 = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", oob_score=True,
            random_state=SEED, n_jobs=-1,
        )
        rf2.fit(Xs, y)
        oob_auc = roc_auc_score(y, rf2.oob_decision_function_[:, 1])
        logger.info("  RF OOB AUC: %.3f", oob_auc)
    except Exception:
        pass
    return pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)


# ── M4: LASSO logistic regression ─────────────────────────────────────────────

def m4_lasso(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegressionCV(
        penalty="l1", solver="liblinear",
        Cs=np.logspace(-3, 1, 15),
        cv=5, scoring="roc_auc",
        class_weight="balanced",
        random_state=SEED, max_iter=1000,
    )
    model.fit(Xs, y)
    logger.info("  LASSO best C=%.4f  non-zero coef: %d/%d",
                model.C_[0], np.sum(model.coef_[0] != 0), X.shape[1])
    coef = np.abs(model.coef_[0])
    return pd.Series(coef, index=X.columns).sort_values(ascending=False)


# ── M5: greedy forward selection ─────────────────────────────────────────────

def m5_forward(
    X: pd.DataFrame,
    y: np.ndarray,
    candidates: list[str],
    max_steps: int = M5_MAX_STEPS,
    min_delta: float = M5_MIN_DELTA,
) -> list[str]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scaler = StandardScaler()
    clf = LogisticRegression(
        C=1.0, solver="lbfgs", class_weight="balanced",
        max_iter=500, random_state=SEED,
    )
    selected: list[str] = []
    remaining = list(candidates)
    best_auc = 0.5

    for step in range(max_steps):
        if not remaining:
            break
        step_best_feat: str | None = None
        step_best_auc = best_auc

        for feat in remaining:
            trial = selected + [feat]
            Xt = scaler.fit_transform(X[trial].values)
            scores = cross_val_score(clf, Xt, y, cv=cv, scoring="roc_auc")
            auc = float(np.mean(scores))
            if auc > step_best_auc:
                step_best_auc = auc
                step_best_feat = feat

        if step_best_feat is None or (step_best_auc - best_auc) < min_delta:
            logger.info("  M5 stopped at step %d (no gain ≥ %.3f)", step + 1, min_delta)
            break

        selected.append(step_best_feat)
        remaining.remove(step_best_feat)
        logger.info("  M5 step %2d: %-45s AUC=%.3f (+%.4f)",
                    step + 1, step_best_feat, step_best_auc, step_best_auc - best_auc)
        best_auc = step_best_auc

    return selected


# ── Borda-count consensus ─────────────────────────────────────────────────────

def borda_consensus(
    rankings: dict[str, pd.Series],
    n_features: int,
    n_top: int = N_TOP,
) -> pd.DataFrame:
    """
    rankings: {method_name: Series(feature -> score, sorted descending)}
    Returns DataFrame with columns = methods + 'borda', rows = top features.
    """
    all_feats = list({f for s in rankings.values() for f in s.index})
    rank_df = pd.DataFrame(index=all_feats)

    for method, scores in rankings.items():
        feat_rank = {f: r + 1 for r, f in enumerate(scores.index)}
        rank_df[method] = [feat_rank.get(f, n_features + 1) for f in all_feats]

    rank_df["borda"] = rank_df.sum(axis=1)
    rank_df = rank_df.sort_values("borda").head(n_top)
    return rank_df


# ── visualisation ─────────────────────────────────────────────────────────────

def plot_heatmap(
    rank_df: pd.DataFrame,
    title: str,
    save_path: Path,
) -> None:
    method_cols = [c for c in rank_df.columns if c != "borda"]
    data = rank_df[method_cols].values.astype(float)

    fig, ax = plt.subplots(figsize=(len(method_cols) * 1.8 + 2, len(rank_df) * 0.45 + 1.5))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r", vmin=1, vmax=N_TOP + 5)
    plt.colorbar(im, ax=ax, label="Rank (lower = better)")

    ax.set_xticks(range(len(method_cols)))
    ax.set_xticklabels(method_cols, fontsize=10)
    ax.set_yticks(range(len(rank_df)))
    ax.set_yticklabels(rank_df.index, fontsize=8)

    for i in range(len(rank_df)):
        for j in range(len(method_cols)):
            r = int(data[i, j])
            txt = str(r) if r <= N_TOP + 1 else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="black" if data[i, j] > 6 else "white")

    ax.set_title(title, fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    logger.info("Saved: %s", save_path)


# ── per (target, source) runner ───────────────────────────────────────────────

def run_one(
    label: str,
    X_raw: pd.DataFrame,
    y: np.ndarray,
    save_path: Path,
) -> dict:
    logger.info("── %s  (n_pos=%d  n_neg=%d  n_feat=%d) ──",
                label, y.sum(), (y == 0).sum(), X_raw.shape[1])

    X = preprocess(X_raw)
    n_feat = X.shape[1]
    logger.info("  after preprocessing: %d features", n_feat)

    # M1
    logger.info("  M1: per-feature AUC …")
    m1 = m1_auc(X, y)

    # M2
    logger.info("  M2: corr-filter + AUC …")
    m2 = m2_corr_auc(X, y)
    logger.info("  M2: %d features kept after corr-filter", len(m2))

    # M3
    logger.info("  M3: Random Forest …")
    m3 = m3_rf(X, y)

    # M4
    logger.info("  M4: LASSO …")
    m4 = m4_lasso(X, y)

    # M5 — candidates = top-M5_CANDIDATES from M2
    candidates = list(m2.head(M5_CANDIDATES).index)
    logger.info("  M5: forward selection on %d candidates …", len(candidates))
    selected = m5_forward(X, y, candidates)
    # Build a ranking Series from forward-selected order
    if selected:
        m5 = pd.Series(
            {f: len(selected) - i for i, f in enumerate(selected)},
        ).sort_values(ascending=False)
        # Append unselected features with score 0 (will get rank n_feat+1 in Borda)
        unselected = pd.Series({f: 0.0 for f in X.columns if f not in m5.index})
        m5 = pd.concat([m5, unselected])
    else:
        m5 = m1.copy() * 0.0   # no features selected → all zero

    rankings = {"M1_AUC": m1, "M2_corrAUC": m2, "M3_RF": m3, "M4_LASSO": m4, "M5_fwd": m5}
    rank_df = borda_consensus(rankings, n_features=n_feat)

    plot_heatmap(rank_df, title=label, save_path=save_path)

    return {
        "rank_df": rank_df,
        "m1": m1, "m2": m2, "m3": m3, "m4": m4,
        "m5_selected": selected,
        "n_features_after_prep": n_feat,
        "n_pos": int(y.sum()),
        "n_neg": int((y == 0).sum()),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── load sources ──────────────────────────────────────────────────────────
    logger.info("Loading sediment source …")
    sed_src = SedimentSource(SED_CSV, name="sediment")
    logger.info("Loading rockchips source …")
    rock_src = SedimentSource(ROCK_CSV, name="rockchips")

    # ── shared background ─────────────────────────────────────────────────────
    logger.info("Generating background (%d points) …", N_TRAIN_BG)
    bg = generate_background(sed_src, N_TRAIN_BG, seed=SEED)
    rock_src.register_samples(bg)

    all_results: dict[str, dict] = {}

    for target, cfg in TARGETS.items():
        logger.info("=" * 60)
        logger.info("TARGET: %s", target)

        instances = load_instances(cfg["sites_csv"], cfg["site_col"], cfg["confidence_map"])
        pos_samples = [inst.sample for inst in instances]
        logger.info("  %d positive instances loaded", len(pos_samples))

        sed_src.register_samples(pos_samples)
        rock_src.register_samples(pos_samples)

        target_results: dict[str, dict] = {}
        for src_name, src in [("sediment", sed_src), ("rockchips", rock_src)]:
            label = f"{target} / {src_name}"
            X_raw, y = collect_xy(pos_samples, bg, src)
            save_path = REPORTS / f"feature_selection_{target}_{src_name}.png"
            res = run_one(label, X_raw, y, save_path)
            target_results[src_name] = res

        all_results[target] = target_results

    # ── write markdown report ─────────────────────────────────────────────────
    lines = [
        "# Feature Selection Report\n",
        "## Methods",
        "| # | Method | Description |",
        "|---|--------|-------------|",
        "| M1 | Per-feature AUC | `max(AUC, 1−AUC)` — direction-agnostic univariate filter |",
        "| M2 | Corr-filter + AUC | Greedy Pearson \|r\|>0.90 removal, then M1 on survivors |",
        "| M3 | Random Forest | Mean-decrease-impurity, balanced class weights, 300 trees |",
        "| M4 | LASSO LR | L1 logistic regression, C tuned by 5-fold CV AUC |",
        "| M5 | Forward selection | 5-fold CV AUC on M2 top-30 shortlist, LogReg probe |",
        "| — | **Borda** | Sum of ranks across M1–M5 (lower = better consensus) |",
        "",
    ]

    for target, target_results in all_results.items():
        lines.append(f"## {target}")
        lines.append("")
        for src_name, res in target_results.items():
            lines.append(f"### {target} / {src_name}")
            lines.append(
                f"Samples: {res['n_pos']} positive, {res['n_neg']} background  "
                f"| Features after preprocessing: {res['n_features_after_prep']}"
            )
            lines.append("")

            # Consensus top-20
            rd = res["rank_df"]
            method_cols = [c for c in rd.columns if c != "borda"]
            header = "| Feature | Borda | " + " | ".join(method_cols) + " |"
            sep    = "|---------|-------|" + "|".join(["-----"] * len(method_cols)) + "|"
            lines += [header, sep]
            for feat, row in rd.iterrows():
                ranks = " | ".join(
                    str(int(row[m])) if row[m] <= N_TOP + 1 else "—"
                    for m in method_cols
                )
                lines.append(f"| `{feat}` | {int(row['borda'])} | {ranks} |")
            lines.append("")

            # M5 forward-selected set
            if res["m5_selected"]:
                lines.append(f"**M5 forward-selected ({len(res['m5_selected'])} features):**")
                lines.append(", ".join(f"`{f}`" for f in res["m5_selected"]))
            else:
                lines.append("**M5:** no features selected (no CV AUC gain above threshold)")
            lines.append("")

    report_path = REPORTS / "feature_selection.md"
    report_path.write_text("\n".join(lines) + "\n")
    logger.info("Saved report: %s", report_path)


if __name__ == "__main__":
    main()
