# MineTrace

**A traceable mineral-prospectivity engine for grounded geoscience agents.**

MineTrace scores any location in Western Australia for nine commodities
(Cu, Au, Ni, W, Sn, Co, Ta, Mn, REE) and — this is the point of the system —
emits, for every score, a structured record from which that score can be
reconstructed exactly. A language-model agent built on top of it can restate
those numbers but cannot invent new ones.

Paper: **https://arxiv.org/pdf/2609.02060**

---

## What this is optimised for, and what it is not

This code is designed for **traceability, not for leaderboard accuracy.**
We state that plainly because the distinction determines whether the system is
the right tool for a given job.

**What is measured and holds** (9 commodities, held-out evaluation — see
`reports/eval_interpretability_summary.json`):

| Property | Result |
|---|---|
| The reported score is exactly reconstructible from its own attribution | **error = 0.0** on 9/9 commodities |
| The expert the narrative names as dominant is the expert whose removal costs the most | **Hit@1 = 0.904**, Hit@2 = 0.981 |
| Every number in the generated narrative resolves to a field of the scoring record | **lookup-failure rate = 0.00** |
| Deleting features in attribution order degrades ranking faster than random order | faithfulness gap **0.156** |
| Attribution stability under bootstrap resampling of the training set | highest among the models we compared |
| A query point missing a whole data modality | the affected experts abstain, weights renormalise, `confidence` falls |

**What this system does not claim.** Gradient-boosted trees and random forests
fit to the same features, the same split and the same seeds are **more accurate**
than this expert tree on every test scenario we ran. If ranking accuracy alone is
the objective, use a random forest. MineTrace exists because a random forest
cannot hand an agent an additive, model-intrinsic decomposition expressed in
named geological hypotheses — and that decomposition is what makes a
language-model narrative auditable.

We regard this trade as the contribution, not as a limitation to be hidden.

---

## How a score is produced

Ten experts, each encoding one geological hypothesis, score a location
independently; the tree combines them by how well each separates known deposits
from background. Full pseudocode: [`reports/algorithm_pseudocode_en.md`](reports/algorithm_pseudocode_en.md).

```
location x
   │
   ├─ φ_s(x)  feature extraction per source ─────────────────────────────────┐
   │     geochemistry  5/10/50 km neighbourhood statistics over 5 assay media │
   │     geophysics    magnetics, gravity, radiometrics, geochronology + grads│
   │     structure     fault / worm distances, densities, lithology, cover    │
   │                                                                          │
   ├─ expert scoring   z = (φ − μ_bg)/σ_bg, direction learned from data,      │
   │                   p = σ( Σ w·z / Σ w )     ← w from domain priors        │
   │                   a source it cannot read ⇒ the expert ABSTAINS          │
   │                                                                          │
   ├─ aggregation      score = Σ_c e_c·p_c / Σ_c e_c,  e_c = AUC-derived      │
   │                   confidence = Σ e_c / Σ W_c   ← how much evidence was   │
   │                                                   actually available     │
   └─ narrative        PointNarrative: score, tier, ranked expert             │
                       contributions, ranked feature signals with z-scores,   │
                       each carrying its source ───────────────────────────────┘
```

Identity satisfied exactly (measured to 0.0 on all nine commodities):

```
score = Σ_c (e_c / Σ e) · σ( Σ_k feature_w[c][k] · feature_z[c][k] )
```

The ten experts: target enrichment, pathfinder association, element
co-enrichment, magnetics, gravity, radiometrics, geochronology, faults,
gradient worms, lithology.

---

## Layout

```
core/                    domain-agnostic engine
  expert_tree.py           CompositeNode: AUC weighting, abstention, confidence
  expert.py                Expert protocol and registry
  belief.py, bk_fusion.py  Dempster-Shafer track (alternative scorer)
  scorer.py                the pluggable-scorer seam
domains/geochem/         Western Australia instantiation
  sources/                 assay, geophysics, geology feature extractors
  experts/                 the ten experts
  prospectivity_model.py   fit / plan / score orchestration
  narrative.py             NodeScore -> PointNarrative (structured, not prose)
scripts/                 evaluation and data preparation
  eval_comprehensive.py    9 commodities x 4 negative-sampling strategies
  eval_interpretability.py additivity, deletion curves, expert-ablation Hit@k
  eval_baselines.py        standard ML baselines on identical splits
  eval_explanation_quality.py  explanation metrics, identical protocol per model
  eval_degradation.py      behaviour under withheld data sources
  eval_confidence.py       selective prediction
  run_all_evals.sh         SLURM driver for the full suite
demo/                    the agent layer of the paper (tools, narrative demos)
reports/                 pseudocode, reproduction guide, evaluation outputs
tests/                   unit tests
```

---

## Getting started

```bash
git clone https://github.com/yihaoding/minetrace
cd minetrace
python -m pip install -e .          # numpy, scipy, scikit-learn, pandas, rasterio, geopandas
```

Score a point once the data is in place:

```python
from core.catalog import DataCatalog
from domains.geochem.prospectivity_model import ProspectivityModel, TargetConfig
from domains.geochem.samples import GeochemSample
from domains.geochem import narrative

cfg = TargetConfig(
    target="Cu",
    pathfinders={"Au": 1.5, "Mo": 1.5, "Ag": 1.0, "Co": 0.8, "Bi": 0.8, "Pb": 0.5},
    ratio_features=["ratio_Cu_Mo", "ratio_Au_Cu"],
    confidence_map={"Mine": 1.0, "Deposit": 0.9},
    sites_csv="datasets/geochemical/states/sites_enriched/enriched_Cu_sites.csv",
)
model = ProspectivityModel(catalog=DataCatalog.from_defaults(), config=cfg)
model.plan(); model.setup()

ns = model.score_batch([GeochemSample(site_code="q1", x=120.5, y=-28.0)])[0]
print(narrative.describe(ns).to_markdown())     # score, tier, experts, feature z-scores
```

`ns` carries the whole tree: `ns.score`, `ns.confidence`, `ns.children`,
`ns.all_feature_signals()`. Nothing downstream needs to re-run scoring to
explain it.

---

## Data

The evaluation uses ~68 GB that this repository does **not** redistribute:

| Source | Provider | Obtain via |
|---|---|---|
| WAMEX geochemistry (stream sediment, rockchip, drillhole, shallow drill, soil) | GSWA / DMIRS, Western Australia | `scripts/download_datasets.sh` |
| Magnetics, gravity, radiometrics, geochronology rasters | GSWA / Geoscience Australia | same |
| Faults, gradient worms, geology, Cenozoic cover | GSWA | same |
| Known-deposit site catalogue | GSWA MINEDEX-derived | same |

Please observe each provider's own licence and attribution requirements; they
are not covered by this repository's MIT licence.

---
## What this is optimised for, and what it is not

This code is designed for **traceability, not for leaderboard accuracy.**
We state that plainly because the distinction determines whether the system is
the right tool for a given job — and we report the numbers that show it.

Every comparison below uses **identical conditions**: the same positives, the
same held-out split, the same seeds, and baselines restricted to the same 195
features the ten experts read. Baselines: a depth-4 decision tree, a
500-tree random forest, XGBoost (400 trees, depth 4), and L1-regularised
logistic regression.

### Performance and attribution stability, side by side

Mean over nine commodities. AUC columns: four negative-sampling strategies.
Attribution stability: how much of a query point's top-3 attribution survives
refitting on a bootstrap resample of the training data (Jaccard, 10 resamples).

| Model | random | far-random | true-nonmine | **spatial** | **attribution stability** |
|---|---|---|---|---|---|
| **Expert tree (this repo)** | 0.906 | 0.930 | 0.800 | 0.673 | **0.631** |
| Decision tree | 0.931 | 0.929 | 0.827 | 0.573 | 0.351 |
| **Random forest** | **0.998** | **0.995** | **0.941** | **0.735** | 0.582 |
| XGBoost | 0.995 | 0.993 | 0.929 | 0.724 | 0.498 |
| L1-logistic | 0.977 | 0.978 | 0.857 | 0.729 | 0.597 |

The first three AUC columns hold out deposits at random; because deposits
cluster and features are 5/10/50 km neighbourhood statistics, they measure
near-mine (brownfield) ranking. `spatial` trains on the south and tests on the
north, 235–622 km apart, and is the only extrapolation test.

### Which one to use

**Use a random forest when the output is a ranked drill list and nothing else
has to be justified.** It is the most accurate model in every one of the four
settings, best in 26 of the 36 commodity × scenario cells against the expert
tree's one, and its vote margin is also the best available guide to which of
its own predictions are wrong (ρ = +0.707 against error, versus −0.049 for our
`confidence`).

**Use the expert tree when the score has to survive being questioned.** Its
explanation is the scoring computation itself rather than a post-hoc
approximation — the reported score reconstructs exactly from its own
attribution (error 0.0 on 9/9 commodities), the expert it names as dominant is
the one whose removal costs the most (Hit@1 0.904), and every number in the
generated narrative resolves to a field of the record (lookup failure 0.00).

**Prefer it in particular when the explanation must describe the deposit rather
than the training sample.** At 0.631 it holds its attribution best of the five
models under bootstrap resampling — the decision tree, whose explanations look
the tidiest at 2.1 features, keeps only 0.351 of its top-3 and is also the
weakest extrapolator at 0.573.

**Prefer it where coverage is uneven.** An expert whose data source is missing
at a query point abstains, the remaining weights renormalise, and `confidence`
falls with the evidence actually available (0.97 → 0.38 as six of seven sources
are withheld); a fixed-width model must impute the missing columns and answers
in the same tone regardless.

**Treat every model with caution for greenfield search.** Under the spatial
split all five fall to 0.57–0.74, and on Mn the expert tree scores below chance
(0.269) because the learned direction inverts in the northern region.

**If you want both, score with the forest and explain with the expert tree.**
Nothing in this repository prevents that pairing, and we have not evaluated it;
it is the obvious next experiment rather than a result.

## Reproducing the evaluation

Everything is seed-fixed. Full details — package versions, every derived seed,
and a table mapping each reported number to the JSON field that holds it — are
in [`reports/REPRODUCE.md`](reports/REPRODUCE.md).

```bash
sbatch scripts/run_all_evals.sh          # SLURM, ~3.5 h
# or a single commodity:
python scripts/eval_baselines.py --metals Cu
```

**Known limitations, stated up front:**

1. Train and test positives are **not spatially separated**. Deposits cluster
   (71% of Cu records lie within 1 km of another record) and features are
   5/10/50 km neighbourhood statistics, so the `random`, `far_random` and
   `true_nonmine` scenarios measure near-mine (brownfield) ranking, not
   discovery in new terrain. The `spatial` scenario (train south, test north,
   235–622 km apart) is the extrapolation test; all models score far lower there.
2. `confidence` reflects **data coverage**, not expected error. It falls
   monotonically as sources are withheld, but does not identify which individual
   predictions are wrong.
3. The REE configuration is incomplete: the assay tables carry individual
   rare-earth columns (Ce, Dy, Er, Eu, …) but no aggregate `REE`, so the REE
   enrichment expert has no target element and runs on pathfinders alone.
4. Mn scores below chance under the spatial split (AUC 0.269), i.e. the learned
   direction inverts in the northern region.

---

## Citation

```bibtex
@article{minetrace2026,
  title  = {MineTrace: Traceable Mineral Prospectivity for Grounded Geoscience Agents},
  author = {Grant, Zhang and others},
  year   = {2026},
  eprint = {2609.02060},
  url    = {https://arxiv.org/pdf/2609.02060}
}
```

## Licence

MIT, see [LICENSE](LICENSE). Data are licensed separately by their providers.
