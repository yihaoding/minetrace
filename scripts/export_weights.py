"""Export fitted model PARAMETERS ONLY (no data) to JSON — one file per metal.

Fits the full 10-expert tree on all known sites, then dumps every fitted
parameter that is a plain scalar/dict: tree-level AUC weights, each expert's
per-source state (bg_mean, bg_std, feature weights, direction, source_weight),
model background element stats, and the target config.

The JSONs are KB-sized and language-agnostic (frontend can read them
directly), but they CANNOT score new points by themselves — scoring needs the
assay/raster/vector data to extract features. Use them for display/transfer;
keep serving on the cluster via scripts/serve_model.py.

Usage (compute node)::

    conda run -n geochem python scripts/export_weights.py [METAL ...]

Defaults to all 9 metals. Output: models/weights/<METAL>_weights.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)

from domains.geochem.geochem_agent import (  # noqa: E402
    GeochemAgent, GeochemTargetConfig, WA_BBOX,
)
from scripts.eval_comprehensive import METAL_CONFIGS, SEED, build_catalog  # noqa: E402

OUT_DIR = ROOT / "models" / "weights"

_PRIMITIVES = (str, int, float, bool, type(None))


def _jsonify(obj, depth: int = 0):
    """Recursively keep JSON-serialisable fitted parameters; drop data blobs."""
    if depth > 6:
        return None
    if isinstance(obj, _PRIMITIVES):
        return obj
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (frozenset, set, tuple, list)):
        vals = [_jsonify(v, depth + 1) for v in obj]
        return [v for v in vals if v is not None or None in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            jv = _jsonify(v, depth + 1)
            if jv is not None or v is None:
                out[str(k)] = jv
        return out
    if hasattr(obj, "__dict__"):
        return _jsonify(vars(obj), depth + 1)
    if hasattr(obj, "__slots__"):  # e.g. _SourceState
        return _jsonify({s: getattr(obj, s) for s in obj.__slots__
                         if hasattr(obj, s)}, depth + 1)
    return None  # numpy arrays, DataFrames, sources, trees … = data, dropped


def export_metal(metal: str) -> Path:
    cfg = METAL_CONFIGS[metal]
    tc = GeochemTargetConfig(
        target=metal,
        pathfinders=cfg["pathfinders"],
        ratio_features=cfg["ratio_features"],
        confidence_map=cfg["confidence_map"],
        sites_csv=cfg["sites_csv"],
    )
    model = GeochemAgent(
        catalog=build_catalog(), target_config=tc,
        n_bg=300, bbox=WA_BBOX, seed=SEED,
    )
    model.plan(check_bbox=WA_BBOX)
    model.setup()

    tree = model._tree
    experts = {}
    for child in tree.children:
        state = {k: _jsonify(v) for k, v in vars(child).items()
                 if not k.startswith("__")}
        experts[child.name] = {k: v for k, v in state.items() if v not in (None, {})}

    doc = {
        "target": metal,
        "seed": SEED,
        "n_fit_positives": len(model._fit_positives),
        "n_fit_background": len(model._fit_background),
        "config": {
            "pathfinders": cfg["pathfinders"],
            "ratio_features": cfg["ratio_features"],
            "confidence_map": cfg["confidence_map"],
        },
        "tree_weights": _jsonify(getattr(tree, "_child_weights", {})),
        "bg_element_mean": _jsonify(model._bg_element_mean),
        "bg_element_std": _jsonify(model._bg_element_std),
        "experts": experts,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{metal}_weights.json"
    out.write_text(json.dumps(doc, indent=2, allow_nan=True))
    print(f"[{metal}] {len(experts)} experts -> {out} "
          f"({out.stat().st_size / 1024:.0f} KB)")
    return out


def main() -> None:
    metals = sys.argv[1:] or list(METAL_CONFIGS)
    for metal in metals:
        if metal not in METAL_CONFIGS:
            print(f"Unknown metal {metal}; choose from {list(METAL_CONFIGS)}")
            continue
        export_metal(metal)


if __name__ == "__main__":
    main()
