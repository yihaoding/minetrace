"""Fit the full 10-expert tree on ALL known sites (production model, no
held-out split) and save it as a single pickle bundle for serving.

The saved bundle scores new points with the "global" Dempster-Shafer belief
fusion (reliability = global AUC weight) — the best scorer in
reports/belief_compare_9x3.log — plus the linear tree score for reference.

Usage (compute node!)::

    conda run -n geochem python scripts/save_fitted_model.py [METAL ...]

Defaults to Cu. Bundles land in models/<METAL>_10expert_global.pkl.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)

from domains.geochem.geochem_agent import (  # noqa: E402
    GeochemAgent, GeochemTargetConfig, WA_BBOX,
)
from domains.geochem.persistence import save_fitted, load_fitted  # noqa: E402
from scripts.eval_comprehensive import METAL_CONFIGS, SEED, build_catalog  # noqa: E402

OUT_DIR = ROOT / "models"


def fit_and_save(metal: str) -> Path:
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
    model.setup()  # no exclude_ids: production fit on all positives
    print(f"[{metal}] fitted {len(model.active_experts())} experts: "
          f"{model.active_experts()}")

    out = save_fitted(model, OUT_DIR / f"{metal}_10expert_global.pkl")
    print(f"[{metal}] saved -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def smoke_test(path: Path, metal: str) -> None:
    """Reload the bundle and score one known positive + one random point."""
    fm = load_fitted(path)
    fmt = lambda v: "None" if v is None else f"{v:.3f}"  # noqa: E731
    pos = fm.model._fit_positives[0]
    r_pos = fm.score_point(pos.x, pos.y, point_id="smoke_pos")
    r_new = fm.score_point(121.0, -27.5, point_id="smoke_new")
    print(f"[{metal}] smoke: known positive ({pos.x:.3f},{pos.y:.3f}) "
          f"global={fmt(r_pos['global'])} tree={fmt(r_pos['tree'])}")
    print(f"[{metal}] smoke: new point (121.0,-27.5) "
          f"global={fmt(r_new['global'])} tree={fmt(r_new['tree'])}")


def main() -> None:
    metals = sys.argv[1:] or ["Cu"]
    for metal in metals:
        if metal not in METAL_CONFIGS:
            print(f"Unknown metal {metal}; choose from {list(METAL_CONFIGS)}")
            continue
        path = fit_and_save(metal)
        smoke_test(path, metal)


if __name__ == "__main__":
    main()
