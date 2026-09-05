"""HTTP scoring service over saved model bundles — for frontend integration.

Loads every models/<METAL>_10expert_global.pkl once at startup (seconds),
then serves point queries in milliseconds. No refitting, ever.

Run (compute node)::

    conda run -n geochem uvicorn scripts.serve_model:app --host 0.0.0.0 --port 8720

Endpoints
---------
GET /health                        -> {"status": "ok", "metals": ["Cu", ...]}
GET /metals                        -> per-metal expert list
GET /score?metal=Cu&x=121.0&y=-27.5
    -> {"metal": "Cu", "x": ..., "y": ...,
        "score_global": 0.63,   # DS belief fusion, global reliability (main)
        "score_tree": 0.61,     # linear AUC-weighted tree (reference)
        "experts": {"cu_enrichment": {"score": ..., "weight": ...}, ...}}
POST /score_batch  {"metal": "Cu", "points": [[x, y], ...]}   (max 500)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from domains.geochem.persistence import load_fitted  # noqa: E402

MODELS_DIR = ROOT / "models"
MAX_BATCH = 500

app = FastAPI(title="GAD prospectivity scoring", version="1.0")
_models: dict = {}


@app.on_event("startup")
def _load_all() -> None:
    for pkl in sorted(MODELS_DIR.glob("*_10expert_global.pkl")):
        metal = pkl.name.split("_")[0]
        _models[metal] = load_fitted(pkl)
        print(f"loaded {metal} <- {pkl.name}")
    if not _models:
        print(f"WARNING: no bundles found in {MODELS_DIR}")


def _get(metal: str):
    fm = _models.get(metal)
    if fm is None:
        raise HTTPException(404, f"No model for '{metal}'. Loaded: {list(_models)}")
    return fm


def _payload(fm, metal: str, x: float, y: float) -> dict:
    res = fm.score_point(x, y)
    node = res["node"]
    experts = {}
    if node is not None:
        for name, child in (node.children or {}).items():
            experts[name] = {
                "score": child.score,
                "weight": (node.weights or {}).get(name),
            }
    return {
        "metal": metal, "x": x, "y": y,
        "score_global": res["global"],
        "score_tree": res["tree"],
        "bel": getattr(node, "bel", None) if node is not None else None,
        "pl": getattr(node, "pl", None) if node is not None else None,
        "experts": experts,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "metals": list(_models)}


@app.get("/metals")
def metals() -> dict:
    return {m: fm.expert_names for m, fm in _models.items()}


@app.get("/score")
def score(metal: str, x: float, y: float) -> dict:
    return _payload(_get(metal), metal, x, y)


class BatchRequest(BaseModel):
    metal: str
    points: list[tuple[float, float]] = Field(max_length=MAX_BATCH)


@app.post("/score_batch")
def score_batch(req: BatchRequest) -> dict:
    fm = _get(req.metal)
    return {"metal": req.metal,
            "results": [_payload(fm, req.metal, x, y) for x, y in req.points]}
