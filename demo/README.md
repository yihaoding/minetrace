# GroundedGeo — EMNLP Demo

LLM tool-use agent for hallucination-free mineral prospectivity reasoning
over Western Australia. Wraps the existing 9-commodity expert-tree
scoring pipeline (in [domains/geochem/](../domains/geochem/)) with a
typed tool layer, a precomputed case-study library, and a Gradio demo UI.

**Target venue:** EMNLP demonstrations track (6-page long-form variant).

---

## Folder layout

```
demo/
├── README.md                 ← this file (project status + plan + TODO)
│
├── paper/                    ← LaTeX source + bibliography
│   ├── acl_latex.tex         ← 6-page draft with Chinese inline TODOs
│   ├── references.csv        ← 30-entry citation list (import → Google Sheets)
│   └── custom.bib            ← (TODO) BibTeX generated from references.csv
│
├── agent/                    ← (TODO) the new agent layer (~600 LOC)
│   ├── __init__.py
│   ├── regions.py            ← Region(bbox/circle) class
│   ├── region_scoring.py     ← two-stage 30×30 → 100×100 grid scoring
│   ├── case_study_loader.py  ← load precomputed JSON
│   ├── model_cache.py        ← lazy LRU cache over 9 ProspectivityModels
│   ├── tools.py              ← 12 typed tools + OpenAI JSON schema
│   ├── system_prompt.py      ← faithfulness constraints
│   └── runner.py             ← OpenAI tool-use loop + audit trail
│
├── scripts/                  ← (TODO) entry points
│   ├── build_case_studies.py ← offline batch: ~100 (region × metal) JSON
│   ├── demo_agent_gradio.py  ← Gradio web UI with map + tool trace
│   └── eval_agent_grounding.py ← condition A vs B evaluation runner
│
├── tests/                    ← (TODO) unit tests for tools (no LLM calls)
│   └── test_agent_tools.py
│
├── case_studies/             ← (TODO) ~100 precomputed JSON
│   ├── yilgarn_au.json
│   ├── pilbara_ni.json
│   └── ...
│
├── eval/                     ← (TODO) evaluation prompt set + results
│   ├── prompts/
│   │   ├── 30_eval_prompts.json   ← author: 2 geology students
│   │   └── tool_call_annotations.json  ← per-prompt expected tools
│   └── results/
│       ├── condition_A_prompt_only.jsonl
│       ├── condition_B_grounded.jsonl
│       ├── numeric_hallucinations.csv
│       └── human_eval_ratings.csv
│
├── figures/                  ← (TODO) paper figures
│   ├── F1_bad_llm_example.pdf
│   ├── F2_architecture.pdf
│   ├── F3_demo_scenarios.pdf
│   └── F4_sxs_comparison.pdf
│
└── sessions/                 ← (TODO) recorded demo dialogues
    ├── scenario_a_regional_recon.json
    ├── scenario_b_tenement.json
    ├── scenario_c_undrilled.json
    └── scenario_d_why_question.json
```

---

## What this layer adds on top of existing code

The scoring engine (Layer 0/1) and the reasoning module
(`narrative.describe()` at Layer 2) already exist in
[`../domains/geochem/`](../domains/geochem/) and stay where they are.

The `demo/` folder houses **only the new agent layer** plus
demo-specific assets (paper, figures, eval data, recorded sessions).
Code in `demo/agent/` imports from the existing project; nothing in
the existing project imports from `demo/`.

---

## Current status (as of 2026-05-18)

| Workstream | Status |
|------------|--------|
| Scoring engine (9 metals × 4 negative strategies) | ✅ done — AUC summary in `paper/acl_latex.tex` Table 1 |
| Narrative module (`PointNarrative`) | ✅ done — `domains/geochem/narrative.py` |
| Paper outline + draft | ✅ done — `paper/acl_latex.tex` (6 pages with placeholders) |
| Citation list | ✅ done — `paper/references.csv` (30 entries) |
| Agent layer code | ⬜ not started |
| Case study library | ⬜ not started |
| Evaluation prompts | ⬜ not started — **needs 2 geology students to author** |
| Human evaluation | ⬜ not started — **needs 3 geology students to rate** |
| HuggingFace Space deployment | ⬜ not started |

---

## System architecture

```
═══════════════════════════════════════════════════════════════════
 Layer 4 │ Demo UI                       Gradio + folium map
═══════════════════════════════════════════════════════════════════
 Layer 3 │ Agent Layer (★ new)           runner.py + tools.py +
         │                                system_prompt.py +
         │                                audit trail per turn
═══════════════════════════════════════════════════════════════════
 Dual    │ Named region ──────► case_study_loader (precomputed JSON)
 path    │ bbox / circle ─────► region_scoring (live two-stage grid)
═══════════════════════════════════════════════════════════════════
 Layer 2 │ Reasoning (existing)          narrative.describe →
         │                                PointNarrative
═══════════════════════════════════════════════════════════════════
 Layer 1 │ Scoring (existing)            ProspectivityModel per metal,
         │                                NodeScore tree output
═══════════════════════════════════════════════════════════════════
 Layer 0 │ Data (existing)               WAMEX sediment/rockchips,
         │                                magnetic/gravity rasters,
         │                                fault vectors, site catalog
═══════════════════════════════════════════════════════════════════
```

**Three design principles** keep numeric hallucinations structurally
absent at the agent layer:

1. Every tool return includes an explicit `audit` field with
   active sources, active experts, and per-feature z-scores.
2. Only `explain_point` and `explain_region` return text, and even
   then it is a structured sequence of typed `FeatureSignal` records.
3. The system prompt requires every numeric claim in the agent's
   response to be quotable verbatim from a tool return in the same
   turn (verified post-hoc by regex extraction).

---

## Tool inventory (12 tools)

| Tool | Inputs | Returns (key fields) |
|------|--------|----------------------|
| `list_targets` | — | commodity list + AUC summary |
| `describe_target` | `target` | pathfinders, ratio features, active layers |
| `score_point` | `target, lon, lat` | g-score, tier, audit |
| `explain_point` | `target, lon, lat, top_n` | top experts, top signals, tier reason |
| `score_with_direct_evidence` | `target, lon, lat, assays` | combined score + evidence expert |
| `multi_target_scan` | `lon, lat, targets` | ranked metals at point |
| `find_neighbors` | `lon, lat, radius_km, kind` | nearby known sites or samples |
| `resolve_region` | named / bbox / circle spec | `Region` object |
| `rank_targets_in_region` | `region, top_k` | ranked metals over region |
| `scan_region` | `region, target` | grid scores, top anomalies |
| `top_anomalies` | `region, target, top_k, exclude_known` | ranked points with explanations |
| `explain_region` | `region, target` | dominant pathfinders + regional narrative |

---

## Case study library (planned ~100 entries)

| Category | Count | Source |
|----------|-------|--------|
| Cratons × 9 commodities | 45 | Yilgarn / Pilbara / Capricorn / Albany-Fraser / Kimberley |
| Mining regions × 9 commodities | 36 | Eastern Goldfields / Murchison / Pilbara Iron / Telfer |
| Multi-commodity regional ranking | 9 | one per region |
| Curated demo scenarios | 10 | analog search, regional comparison, etc. |
| **Total** | **~100** | |

Region polygons are convex hulls over WAMEX sediment points carrying
the corresponding `CRATON` attribute — no external shapefile licensing
required.

---

## Demo scenarios (paper §5)

| # | Scenario | Example query | Key tool chain |
|---|----------|---------------|----------------|
| (a) | Regional reconnaissance | "Yilgarn 哪种金属最有戏？" | `rank_targets_in_region` → `top_anomalies` |
| (b) | Tenement evaluation | "这个 bbox 应该钻什么？" | `resolve_region(bbox)` → `scan_region` → `top_anomalies` |
| (c) | Targeted undrilled query | "Pilbara 北部 Au > 0.7 且未被钻过的异常" | `top_anomalies(exclude_known=True)` |
| (d) | Regional why-question | "为什么 Eastern Goldfields Au 这么好？" | `explain_region` |

---

## TODO checklist (by priority, by stream)

### 🚩 START IMMEDIATELY (external dependencies, longest lead time)

- [ ] Find 3 geology graduate students for plausibility rating
      (paper §6 main experiment — without this, no human eval)
- [ ] Find 2 geology students to author 30 evaluation prompts
      (can start now, doesn't depend on code being done)

### ① Code & infrastructure (~6 days)

- [ ] Install dependencies: `openai`, `gradio`, `folium`, `shapely`
- [ ] `agent/regions.py` — Region class (bbox + circle)
- [ ] `agent/region_scoring.py` — two-stage grid scoring
- [ ] `agent/case_study_loader.py`
- [ ] `agent/model_cache.py` — lazy LRU over 9 models
- [ ] `agent/tools.py` — 12 tool implementations + OpenAI schema
- [ ] `agent/system_prompt.py` — faithfulness constraints
- [ ] `agent/runner.py` — tool-use loop + audit capture
- [ ] `scripts/demo_agent_gradio.py` — Gradio + folium UI
- [ ] `tests/test_agent_tools.py` — unit tests for all 12 tools

### ② Case study library (~1.75 days)

- [ ] `scripts/build_case_studies.py` — batch generation script
- [ ] Run to produce ~100 JSON files under `case_studies/`
- [ ] Spot-check 10 cases manually

### ③ Evaluation (~4 days + external annotators)

- [ ] Author 30 evaluation prompts (with geology students)
- [ ] `scripts/eval_agent_grounding.py` runs both conditions
- [ ] Numeric hallucination extraction script (regex + verify)
- [ ] Tool-call accuracy annotation (per-prompt expected tool set)
- [ ] Side-by-side HTML interface for human plausibility rating
- [ ] Recruit 3 raters, collect Likert + pairwise preference
- [ ] Compute Krippendorff's α + Wilcoxon p-values
- [ ] Three ablations: no-audit, no-prompt, no-case-study

### ④ Figures (~1.5 days)

- [ ] F1: prompt-only GPT-4o reply with red overlay vs real data
- [ ] F2: 5-layer architecture diagram (TikZ or Inkscape)
- [ ] F3: 4-scenario demo dialogue collage
- [ ] F4: side-by-side response comparison with overlays

### ⑤ Paper finalization (~3 days + supervisor review)

- [ ] Fill XX placeholders with actual evaluation numbers
- [ ] Convert `references.csv` → `paper/custom.bib`
- [ ] Verify 3 flagged citations (Carranza year, Zuo paper, Talebi)
- [ ] Flesh out Intro paragraph 2-3 with real prose
- [ ] Flesh out Related Work with one-sentence-per-cite summaries
- [ ] Compile + check page overflow at 6 pages
- [ ] Internal review (supervisor / second author)

### ⑥ Reproducibility & deployment (~2.5 days)

- [ ] Anonymous GitHub repo (anonymous.4open.science)
- [ ] Top-level README with full setup instructions
- [ ] LICENSE (MIT recommended)
- [ ] Publish all 30 eval prompts + tool annotations
- [ ] Publish all ~100 case study JSON
- [ ] HuggingFace Space deployment (so reviewers can play) ★ killer feature
- [ ] Anonymized 2–3 min demo screencast

---

## Estimated timeline

```
Week 1   Code (S0–S2)              Find annotators     Author 30 prompts
Week 2   Case studies + Gradio UI  Annotators trained  
Week 3   Run evaluation            Collect ratings     Fill paper
Week 4   Compile + review + ship
```

**Earliest possible submission: ~3 weeks** if no blocking external dependency stalls.

---

## Key risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Can't recruit 3 geology raters | High | Start asking now; have backup plan (1 expert + 2 generalist coders trained with rubric) |
| WAMEX republication license | Medium | Verify before pushing case studies to public repo |
| Case study JSON sizes balloon | Low | Limit top-anomalies list to 50; gzip if needed |
| HF Space cold-start latency | Low | Warmup hook on Space start; pre-load top-3 metals |
| Paper overflows 6 pages | Medium | Push appendix-eligible content (full tool schemas, prompt list) out at compile time |

---

## Style & language conventions

- All code, identifiers, file names: **English**
- All inline LaTeX comments in `acl_latex.tex`: **Chinese** (per author preference)
- All paper prose: **English**
- All conversational discussion (this README sections that are 中文): mix is fine

---

## How to pick this back up next session

1. Read this README top to bottom
2. Check the "Current status" table for what's done
3. Pick the next unchecked item from the "🚩 START IMMEDIATELY" or stream-① block
4. The `paper/acl_latex.tex` Chinese inline comments tell you what each section needs

The README is the single source of truth for the demo project state.
Update it as workstreams move from ⬜ to ✅.
