# Algorithms

Implementation: `core/expert_tree.py`, `domains/geochem/experts/*.py`,
`domains/geochem/sources/*.py`, `domains/geochem/prospectivity_model.py`,
`domains/geochem/narrative.py`.

**Notation.** `x` — query location; `S` — active data sources; `E` — experts;
`P` — known-deposit (positive) samples; `B` — background samples;
`φ_s(x)` — feature map of source `s`; `μ, σ` — background mean and standard
deviation; `w` — feature weight (domain prior); `d ∈ {−1,+1}` — learned
direction; `ω_s` — source weight; `W_c` — expert weight; `σ(·)` — logistic
function.

---

## Algorithm 1 Feature extraction — `φ_s(x)`

```
Input : location x, data source s
Output: feature map f

if s is a geochemical assay source then                # 5 media: stream sediment,
    for r ∈ {5, 10, 50} km do                          # rockchip, drillhole,
        N_r ← KDTree(s).query_ball(x, r)               # shallow drill, soil
        for each element e do
            v ← log1p(concentrations of e in N_r), discard non-finite and negative
            if |v| < MIN_SAMPLES then f[·] ← NaN
            else
                f[e_{r}km_median]     ← median(v)
                f[e_{r}km_p90]        ← percentile(v, 90)
                f[e_{r}km_cv]         ← std(v) / mean(v)
                f[e_{r}km_frac_above] ← mean(v > θ_e)  # θ_e: 75th pct of e over WA
        f[n_samples_{r}km] ← |N_r|
    for each element e do
        f[e_contrast_5_50]  ← f[e_5km_median]  − f[e_50km_median]
        f[e_contrast_10_50] ← f[e_10km_median] − f[e_50km_median]
        # a difference in log space is the log of a concentration ratio:
        # local enrichment relative to the regional background
    for each configured pair (a,b) do
        f[ratio_a_b] ← f[a_10km_median] − f[b_10km_median]
    f[spatial_coherence_10km] ← mean pairwise distance of the 5 highest values
    f[PC{i}_{5,10}km_median]  ← median of neighbourhood PCA scores

else if s is a geophysical raster source then
    for name ∈ {mag, grav, K, Th, U, LuHf, SmNd} do
        f[name]      ← raster sampled at x
        f[name_grad] ← sqrt(dx² + dy²)                 # horizontal gradient magnitude
    f[K_Th], f[Th_U], f[U_K] ← corresponding ratios

else if s is a structural / geological vector source then
    f[dist_fault_km], f[fault_density_5km], f[fault_density_10km]
    f[dist_worm_mag_km], f[dist_worm_grav_km]
    f[is_rt_*] (lithology one-hot), f[age_ma], f[is_yilgarn], f[is_pilbara], f[is_covered]

return f
```

---

## Algorithm 2 Leaf expert — fit and score

```
procedure EXPERT.FIT(P, B)
    for each source s ∈ required_sources do
        F_P ← {φ_s(x) : x ∈ P};  F_B ← {φ_s(x) : x ∈ B}
        if F_B = ∅ then continue
        for each feature k do
            μ_s[k], σ_s[k] ← mean, std of F_B[k]       # σ < 1e−9 ⇒ set to 1
            w_s[k] ← domain prior                      # equal for enrichment experts;
                                                       # pathfinder prior dict otherwise
            d_s[k] ← +1 if median(F_P[k]) ≥ median(F_B[k]) else −1     # learned from data
        a ← AUC( score of source s alone on P ∪ B )
        if a < 0.5 then a ← 1 − a;  d_s ← −d_s         # source is anti-correlated
        ω_s ← max(0, (a − 0.5) × 2)                    # source weight
    end for

function EXPERT.SCORE(x)
    for each source s with ω_s > 0 do
        f ← φ_s(x);  if f = ⊥ then continue
        z_s[k] ← (f[k] − μ_s[k]) / σ_s[k] · d_s[k]     # direction folded into z
        raw_s  ← Σ_k w_s[k]·z_s[k] / Σ_k w_s[k]        # finite k only: a NaN feature
                                                       # enters neither numerator nor
                                                       # denominator
        if no finite term then continue
        p_s ← σ(raw_s)
    if no source usable then return ⊥                  # ABSTAIN: the expert leaves
                                                       # the aggregation entirely
    score ← Σ_s ω_s·p_s / Σ_s ω_s
    # chain-normalised per-feature weights, so that Σ w·z reconstructs raw_s
    W_feat["s:k"] ← (ω_s / Σ ω) · ( w_s[k] / Σ_{k' present in s} w_s[k'] )
    return NodeScore(score, feature_z = z, feature_w = W_feat, weights = ω)
```

---

## Algorithm 3 Expert tree — fit and score

```
procedure COMPOSITENODE.FIT(P, B)
    for each child expert c ∈ E do
        c.FIT(P, B)
        v ← [ c.SCORE(x).score : x ∈ P ∪ B, score ≠ ⊥ ]
        if |v| < 10 or v has a single class then W_c ← 0; continue
        a ← AUC(v, labels)
        if a < 0.5 then a ← 1 − a;  c.invert ← true    # each node owns exactly one
                                                       # flip of its own output;
                                                       # no ancestor re-applies it
        W_c ← max(0, (a − 0.5) × 2)                    # global weight

function COMPOSITENODE.SCORE(x)
    C ← { c ∈ E : c.SCORE(x) ≠ ⊥ }                     # abstaining experts vanish here
    if C = ∅ then return ⊥
    for c ∈ C do  e_c ← W_c · c.local_reliability(x)   # default reliability 1.0
    if Σ e_c < 1e−9 then
        score ← mean{ c.score : c ∈ C };  confidence ← 0
    else
        score      ← Σ_c e_c · c.score / Σ_c e_c
        confidence ← Σ_{c ∈ C} e_c / Σ_{c ∈ E} W_c     # fraction of fitted weight used
    if self.invert then score ← 1 − score
    return NodeScore(score, confidence, children = C, weights = e)
```

---

## Algorithm 4 End-to-end scoring and narrative

```
procedure FIT(catalog, config, exclude_ids)
    S ← sources passing the coverage probe                # 40 random points, ≥3 samples
                                                          # within 10 km, ≥10% hit rate
    P ← known deposits(config.sites_csv) \ exclude_ids
    B ← uniform random points in the WA bounding box with ≥3 samples within 10 km,
        |B| = 300, fixed seed
    E ← 3 geochemical experts (across all assay sources)
        ∪ 4 geophysical experts ∪ 3 structural experts
    tree ← COMPOSITENODE(E);  tree.FIT(P, B)

function EXPLAIN(x)
    ns ← tree.SCORE(x)
    tier ← Strong if ns.score ≥ 0.80, Moderate if ≥ 0.65, Weak if ≥ 0.52, else Background
    for c ∈ ns.children do
        contrib_c ← (e_c / Σ e) · | ns.children[c].score − 0.5 |
    for each feature key "s:k" do
        attribution["s:k"] ← (e_c / Σ e) · feature_w["s:k"] · feature_z["s:k"]
    return PointNarrative(score, tier, confidence,
                          ranked contrib, ranked attribution)
    # a structured record, not a string: the agent layer may restate its numbers
    # but cannot introduce new ones
```

**Identity (measured: exact to 0.0 on 9/9 commodities).**

```
ns.score = Σ_c (e_c / Σ e) · σ( Σ_k feature_w[c][k] · feature_z[c][k] )
```

---

## Algorithm 5 Evaluation protocol

```
for each commodity do
    all_pos  ← records whose SITE_CODE field matches "Mine|Deposit"
    test_pos ← first 30 of a fixed permutation (seed 42);  train_pos ← remainder
    forbid   ← coordinates of test_pos, in km

    # training set, shared by every model
    train_neg ← random WA points with assay coverage, ≥5 km from any test positive
                (n = 300, seed 142)

    # four test-negative strategies
    random       ← random WA points, covered, ≥5 km from test positives  (n=200, seed 43)
    far_random   ← as above and ≥50 km from any known deposit            (n=200, seed 44)
    true_nonmine ← no_<el>_sites.csv: real deposits of other commodities
                   where the target commodity is absent                  (n≤200, seed 45)
    spatial      ← train on south (Y < −25), test on north (Y ≥ −25);
                   negatives are non-deposit sites in the north          (n=200, seed 47)

    expert_tree ← FIT(train_pos, train_neg)                              # Algorithm 4
    X ← feature table over all active sources, columns named "source:feature"
    X ← restrict to the 195 columns the 10 experts actually read (expert_matched);
        impute NaN with the training median
    for m ∈ {DecisionTree(depth 4), RandomForest(500), XGBoost(400, depth 4),
             L1-Logistic} do  m.fit(X_train, y_train)

    report AUC and P@{10,20,50} for every scenario
```

---

## Algorithm 6 Explanation quality — identical protocol for all models

```
# attribution: each model supplies its own; none is imposed on another
A[expert_tree]   ← (e_c / Σ e) · | feature_w · feature_z |
A[decision_tree] ← weighted impurity decrease of the nodes on the decision path
A[random_forest] ← TreeSHAP (shap.TreeExplainer)
A[xgboost]       ← booster.predict(pred_contribs = True)          # exact TreeSHAP
A[logit_l1]      ← | coef · (x − training mean) |

# faithfulness: claimed top driver vs. the feature that actually moves the score
for each feature j do                                             # all 195, no sampling
    E[i,j] ← | score(x_i) − score(x_i with feature j set to the training median) |
    # for the expert tree this passes through the real model: overwrite the cached
    # source feature, re-score, restore
Hit@1 ← mean_i [ argmax_j A[i,j] = argmax_j E[i,j] ]
ρ     ← mean_i spearman( A[i,·], E[i,·] )
# group level: partition the columns by the expert that reads them, mask whole
# groups, recompute both quantities

# stability
for b = 1 … 10 do
    resample the training set with replacement, refit, recompute A_b
    J_b ← mean_i jaccard( top3(A[i,·]), top3(A_b[i,·]) )
stability ← mean_b J_b
```

---

## Algorithm 7 Missing modalities and self-reported reliability

```
for k = 0 … |S| − 1 do
    for rep = 1 … N_REP do
        D ← k sources drawn at random
        # expert tree: blank D at the test points ⇒ the affected experts return ⊥
        #              and drop out; the remaining weights renormalise
        score_expert, confidence ← tree.SCORE(test points with D blanked)
        # fixed-width models cannot skip a modality; the columns must be imputed
        score_ml ← m.predict(X with columns of D ← training median)
                                                     # XGBoost also run with native NaN
    record AUC(k), mean confidence(k), number of active experts(k)

# selective prediction: each model ranked by its own reliability signal
signal[expert_tree] ← confidence                     # independent of the score
signal[others]      ← | p − 0.5 |                    # a transform of the score;
                                                     # RF additionally: std over trees
for coverage ∈ {100%, 75%, 50%, 25%} do
    retain the highest-signal fraction of points and recompute AUC
ρ_err ← spearman( signal, −| y − p | )
```
