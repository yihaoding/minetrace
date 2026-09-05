"""
Expert Framework — Training-Free Aggregation
Cu Prospectivity — GSWA Stream Sediment

Architecture:
  5 Experts (single-element, correlation, PCA, clustering, causal)
  Each expert: fit(near, back) → score(point) → knowledge_entry()
  Aggregation (training-free):
    - Equal mean
    - Z-score mean  (normalise per-expert score distribution)
    - Borda count   (rank-based voting)
"""

import json
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
from causalnex.structure.notears import from_pandas

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
CU_SITES = '/group/pmc050/yding/gad_reasoning/datasets/geochemical/states/sites/gswa_cu_sites.csv'
SEDIMENT = '/group/pmc050/yding/gad_reasoning/datasets/geochemical/states/state_geochemical/gswa_all_sediment.csv'
NEAR_KM, FAR_KM = 1, 30
SEED = 42
np.random.seed(SEED)

ELEMENTS = ['Cu_ppm', 'Mo_ppm', 'Co_ppm', 'Au_ppm',
            'As_ppm', 'Mn_ppm', 'Fe_ppm', 'Pb_ppm', 'Zn_ppm']

DEG_KM   = 111.0
LON_KM   = DEG_KM * np.cos(np.radians(28))


# ── Helpers ───────────────────────────────────────────────────────────────────
def log_scale(df, scaler=None):
    """Log10-transform and StandardScale a DataFrame; fit scaler if not provided."""
    X = np.log10(df.clip(lower=1e-9).values)
    if scaler is None:
        scaler = StandardScaler()
        return scaler.fit_transform(X), scaler
    return scaler.transform(X), scaler


def _log_val(point, elem):
    return np.log10(max(float(point.get(elem, 1e-9)), 1e-9))


# ── 1. Load & Label ───────────────────────────────────────────────────────────
print("=== 1. Loading data ===")
cu_sites = pd.read_csv(CU_SITES).dropna(subset=['X', 'Y'])
sed = pd.read_csv(SEDIMENT)
sed.replace(-9999, np.nan, inplace=True)
for c in sed.select_dtypes(include=[np.number]).columns:
    if c not in ('X', 'Y'):
        sed.loc[sed[c] < 0, c] = np.nan
sed = sed.dropna(subset=['X', 'Y'])

def to_km(lon, lat):
    return np.column_stack([lon * LON_KM, lat * DEG_KM])

tree = cKDTree(to_km(cu_sites['X'].values, cu_sites['Y'].values))
dist, _ = tree.query(to_km(sed['X'].values, sed['Y'].values), k=1)
sed['dist_km'] = dist
sed['label'] = 'buffer'
sed.loc[dist <= NEAR_KM, 'label'] = 'near'
sed.loc[dist > FAR_KM,  'label'] = 'back'

ELEMENTS = [e for e in ELEMENTS if e in sed.columns]
near = sed[sed['label'] == 'near'][ELEMENTS].copy()
back = sed[sed['label'] == 'back'][ELEMENTS].copy()
print(f"  Near: {len(near)}  |  Background: {len(back)}")
print(f"  Elements ({len(ELEMENTS)}): {ELEMENTS}")

def impute_median(df):
    return df.fillna(df.median())

near_imp = impute_median(near)
back_imp = impute_median(back)


# ── 2. Expert Base Class ──────────────────────────────────────────────────────
class Expert:
    name = "base"
    def fit(self, near, back): raise NotImplementedError
    def score(self, point: pd.Series) -> float: raise NotImplementedError
    def knowledge_entry(self) -> dict: raise NotImplementedError


# ── 3. Expert 1: Single-Element ───────────────────────────────────────────────
class SingleElementExpert(Expert):
    """
    Weighted log-ratio of observed value vs background P90.
    Weight = |fold_change - 1|. Direction learned from data.
    """
    name = "single_element"

    def fit(self, near, back):
        self.stats = {}
        for e in near.columns:
            nv = near[e].dropna()
            bv = back[e].dropna()
            if len(nv) < 10 or len(bv) < 10:
                continue
            back_med = bv.median()
            near_med = nv.median()
            fold = near_med / back_med if back_med > 0 else 1.0
            self.stats[e] = {
                'p90':      float(bv.quantile(0.9)),
                'p10':      float(max(bv.quantile(0.1), 1e-9)),
                'back_med': float(back_med),
                'near_med': float(near_med),
                'fold':     float(fold),
                'direction': 1 if fold >= 1 else -1,
            }
        return self

    def score(self, point):
        s = w = 0.0
        for e, info in self.stats.items():
            v = point.get(e, np.nan)
            if pd.isna(v) or v <= 0:
                continue
            wt = max(abs(info['fold'] - 1), 0.01)
            ref = info['p90'] if info['direction'] == 1 else info['p10']
            denom = ref if info['direction'] == 1 else float(v)
            numer = float(v) if info['direction'] == 1 else ref
            c = wt * np.log10(max(numer, 1e-9) / max(denom, 1e-9))
            if np.isfinite(c):
                s += c; w += wt
        return s / w if w > 0 else 0.0

    def knowledge_entry(self):
        top = sorted(self.stats.items(), key=lambda x: -abs(x[1]['fold'] - 1))[:5]
        return {
            'expert': self.name,
            'top_elements': [
                {'element': e, 'fold': round(v['fold'], 2),
                 'direction': '+' if v['direction'] == 1 else '-',
                 'p90': v['p90']}
                for e, v in top
            ]
        }


# ── 4. Expert 2: Correlation Structure ───────────────────────────────────────
class CorrelationExpert(Expert):
    """
    Δr = r_near - r_back: element-pair correlation difference.
    Score = mean over top pairs of (Δr × log_ratio_e1 × log_ratio_e2).
    Positive = element pairs co-enrich in a near-deposit-consistent way.
    """
    name = "correlation"

    def fit(self, near, back):
        self.back_med = back.median().to_dict()

        def log_df(df):
            return np.log10(df.clip(lower=1e-9))

        near_corr = log_df(near).corr()
        back_corr = log_df(back).corr()
        delta_r   = near_corr - back_corr

        pairs = []
        for e1, e2 in combinations(near.columns, 2):
            dr = float(delta_r.loc[e1, e2])
            if np.isfinite(dr):
                pairs.append((e1, e2, dr))

        pairs.sort(key=lambda x: -abs(x[2]))
        self.top_pairs = pairs[:10]
        return self

    def score(self, point):
        contribs = []
        for e1, e2, dr in self.top_pairs:
            v1 = point.get(e1, np.nan)
            v2 = point.get(e2, np.nan)
            if pd.isna(v1) or pd.isna(v2) or v1 <= 0 or v2 <= 0:
                continue
            m1 = max(self.back_med.get(e1, 1.0), 1e-9)
            m2 = max(self.back_med.get(e2, 1.0), 1e-9)
            c = dr * np.log10(float(v1) / m1) * np.log10(float(v2) / m2)
            if np.isfinite(c):
                contribs.append(c)
        return float(np.mean(contribs)) if contribs else 0.0

    def knowledge_entry(self):
        return {
            'expert': self.name,
            'top_pairs': [
                {'pair': f"{e1} & {e2}", 'delta_r': round(dr, 3),
                 'interpretation': 'co-enrich near deposit' if dr > 0 else 'anti-correlated near deposit'}
                for e1, e2, dr in self.top_pairs[:5]
            ]
        }


# ── 5. Expert 3: PCA ─────────────────────────────────────────────────────────
class PCAExpert(Expert):
    """
    Score = dist(background centroid) - dist(near centroid) in PC1-3 space.
    Positive = closer to near-deposit cluster.
    """
    name = "pca"

    def fit(self, near, back):
        self.elements = near.columns.tolist()
        combined = pd.concat([near, back])
        labels_arr = np.array([1] * len(near) + [0] * len(back))

        X_sc, self.scaler = log_scale(combined)

        self.pca = PCA(n_components=min(3, len(self.elements)))
        X_pc = self.pca.fit_transform(X_sc)

        self.near_centroid = X_pc[labels_arr == 1].mean(axis=0)
        self.back_centroid = X_pc[labels_arr == 0].mean(axis=0)
        self.loadings = pd.DataFrame(
            self.pca.components_.T,
            index=self.elements,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)]
        )
        return self

    def score(self, point):
        vals = [_log_val(point, e) for e in self.elements]
        x_sc = self.scaler.transform(np.array(vals).reshape(1, -1))
        x_pc = self.pca.transform(x_sc)[0]
        return float(np.linalg.norm(x_pc - self.back_centroid) -
                     np.linalg.norm(x_pc - self.near_centroid))

    def knowledge_entry(self):
        ev = self.pca.explained_variance_ratio_
        top_pc1 = self.loadings['PC1'].abs().nlargest(3).index.tolist()
        return {
            'expert': self.name,
            'explained_variance_pct': [round(v * 100, 1) for v in ev],
            'pc1_drivers': top_pc1,
        }


# ── 6. Expert 4: Clustering (GMM) ────────────────────────────────────────────
class ClusteringExpert(Expert):
    """
    Score = P(near component) - P(back component) under a 2-component GMM.
    """
    name = "clustering"

    def fit(self, near, back):
        self.elements = near.columns.tolist()
        combined = pd.concat([near, back])
        labels_arr = np.array([1] * len(near) + [0] * len(back))

        X_sc, self.scaler = log_scale(combined)

        self.gmm = GaussianMixture(n_components=2, random_state=SEED, n_init=3)
        self.gmm.fit(X_sc)

        comp = self.gmm.predict(X_sc)
        p0 = labels_arr[comp == 0].mean()
        p1 = labels_arr[comp == 1].mean()
        self.near_comp  = 0 if p0 >= p1 else 1
        self.near_purity = max(p0, p1)
        return self

    def score(self, point):
        vals = [_log_val(point, e) for e in self.elements]
        x_sc  = self.scaler.transform(np.array(vals).reshape(1, -1))
        probs = self.gmm.predict_proba(x_sc)[0]
        return float(probs[self.near_comp] - probs[1 - self.near_comp])

    def knowledge_entry(self):
        return {
            'expert': self.name,
            'near_deposit_component': self.near_comp,
            'near_component_purity': round(self.near_purity, 3),
        }


# ── 7. Expert 5: Causal (NOTEARS) ────────────────────────────────────────────
class CausalExpert(Expert):
    """
    ΔW = W_near - W_back: directed edges that strengthen near deposits.
    Score = mean over top |ΔW| edges of (ΔW × log_ratio(parent) × log_ratio(child)).
    """
    name = "causal"
    MAX_SAMPLES = 500

    def fit(self, near, back):
        self.elements = near.columns.tolist()
        self.back_med = back.median().to_dict()

        def subsample_scale(df):
            sub = df.sample(min(len(df), self.MAX_SAMPLES), random_state=SEED)
            X_sc, _ = log_scale(sub)
            return pd.DataFrame(X_sc, columns=df.columns)

        near_sc = subsample_scale(near)
        back_sc = subsample_scale(back)

        sm_near = from_pandas(near_sc, w_threshold=0.1, tabu_edges=[], tabu_parent_nodes=[])
        sm_back = from_pandas(back_sc, w_threshold=0.1, tabu_edges=[], tabu_parent_nodes=[])

        nodes   = self.elements
        W_near  = nx.to_numpy_array(sm_near, nodelist=nodes)
        W_back  = nx.to_numpy_array(sm_back, nodelist=nodes)
        delta_W = W_near - W_back

        edges = [
            (nodes[i], nodes[j], float(delta_W[i, j]))
            for i in range(len(nodes))
            for j in range(len(nodes))
            if i != j and abs(delta_W[i, j]) > 0.05
        ]
        edges.sort(key=lambda x: -abs(x[2]))
        self.top_edges = edges[:10]
        return self

    def score(self, point):
        contribs = []
        for ep, ec, dw in self.top_edges:
            vp = point.get(ep, np.nan)
            vc = point.get(ec, np.nan)
            if pd.isna(vp) or pd.isna(vc) or vp <= 0 or vc <= 0:
                continue
            mp = max(self.back_med.get(ep, 1.0), 1e-9)
            mc = max(self.back_med.get(ec, 1.0), 1e-9)
            c  = dw * np.log10(float(vp) / mp) * np.log10(float(vc) / mc)
            if np.isfinite(c):
                contribs.append(c)
        return float(np.mean(contribs)) if contribs else 0.0

    def knowledge_entry(self):
        return {
            'expert': self.name,
            'top_causal_edges': [
                {'edge': f'{ep} → {ec}',
                 'delta_W': round(dw, 3),
                 'interpretation': 'stronger causal link near deposit' if dw > 0 else 'weaker near deposit'}
                for ep, ec, dw in self.top_edges[:5]
            ]
        }


# ── 8. Fit All Experts ────────────────────────────────────────────────────────
print("\n=== 2. Training experts ===")
experts = [
    SingleElementExpert(),
    CorrelationExpert(),
    PCAExpert(),
    ClusteringExpert(),
    CausalExpert(),
]
for exp in experts:
    exp.fit(near_imp, back_imp)
    print(f"  ✓ {exp.name}")

print("\n--- Knowledge entries ---")
for exp in experts:
    print(f"\n[{exp.name}]")
    print(json.dumps(exp.knowledge_entry(), indent=2, ensure_ascii=False))


# ── 9. Individual Expert AUC (100+100) ───────────────────────────────────────
print("\n=== 3. Individual expert AUC (100 pos + 100 neg) ===")
pos100 = near_imp.sample(100, random_state=SEED)
neg100 = back_imp.sample(100, random_state=SEED)
eval_df = pd.concat([
    pos100.assign(_label=1),
    neg100.assign(_label=0)
], ignore_index=True)

labels = eval_df['_label'].values
expert_scores_arr = {}
expert_auc = {}
for exp in experts:
    scores = eval_df[ELEMENTS].apply(exp.score, axis=1).to_numpy()
    expert_scores_arr[exp.name] = scores
    expert_auc[exp.name] = roc_auc_score(labels, scores)
    print(f"  {exp.name:20s}: AUC = {expert_auc[exp.name]:.4f}")


# ── 10. Training-Free Aggregation ────────────────────────────────────────────
print("\n=== 4. Training-Free Aggregation ===")

score_matrix = np.stack(
    [expert_scores_arr[e.name] for e in experts], axis=1
)  # (n_samples, n_experts)

agg_mean   = score_matrix.mean(axis=1)

z_matrix   = (score_matrix - score_matrix.mean(axis=0)) / (score_matrix.std(axis=0) + 1e-9)
agg_zscore = z_matrix.mean(axis=1)

rank_matrix = np.argsort(np.argsort(score_matrix, axis=0), axis=0).astype(float)
agg_borda   = rank_matrix.mean(axis=1)

aggregations = {
    'equal_mean':  agg_mean,
    'zscore_mean': agg_zscore,
    'borda_count': agg_borda,
}
agg_auc = {name: roc_auc_score(labels, s) for name, s in aggregations.items()}

for name, auc in agg_auc.items():
    print(f"  {name:20s}: AUC = {auc:.4f}")


# ── 11. Visualisation ─────────────────────────────────────────────────────────
print("\n=== 5. Visualisation ===")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Expert Framework — Training-Free Aggregation | GSWA Sediment (Cu)',
             fontsize=13, fontweight='bold')

ax = axes[0]
bar_names  = [e.name.replace('_', '\n') for e in experts] + \
             [n.replace('_', '\n') for n in aggregations]
bar_vals   = [expert_auc[e.name] for e in experts] + list(agg_auc.values())
bar_colors = ['#4A90D9'] * len(experts) + ['#E63946'] * len(aggregations)
ax.barh(bar_names, bar_vals, color=bar_colors, alpha=0.85)
ax.axvline(0.5, color='gray', linestyle='--', lw=1)
for i, v in enumerate(bar_vals):
    ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)
ax.set_xlabel('AUC')
ax.set_title('AUC: Individual Experts vs Aggregations')
ax.set_xlim(0.4, 1.0)

ax = axes[1]
bins = np.linspace(agg_zscore.min(), agg_zscore.max(), 30)
ax.hist(agg_zscore[labels == 0], bins=bins, alpha=0.6, color='#AAAAAA',
        density=True, label='Background')
ax.hist(agg_zscore[labels == 1], bins=bins, alpha=0.6, color='#E63946',
        density=True, label='Near-deposit')
ax.axvline(0, color='black', linestyle='--', lw=1)
ax.set_xlabel('Z-score mean')
ax.set_ylabel('Density')
ax.set_title('Score Distribution (Z-score Mean)')
ax.legend()

ax = axes[2]
corr = pd.DataFrame(score_matrix,
                    columns=[e.name for e in experts]).corr()
sns.heatmap(corr, ax=ax, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', square=True, linewidths=0.5,
            annot_kws={'size': 9})
ax.set_title('Expert Score Correlation')

plt.tight_layout()
out = '/group/pmc050/yding/gad_reasoning/notebooks/expert_results.png'
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.show()
print(f"Figure saved → {out}")
print("\nDone.")
