"""ElementCorrelationExpert — element pair co-enrichment near deposits.

Conceptual basis (from original 02_expert_rl_sediment.py CorrelationExpert):
  Near a deposit, specific element pairs co-enrich (Δr > 0).
  This expert captures that signal at the neighbourhood level.

Mechanism:
  For each element pair (A, B), the co-enrichment score at a point is:
    co(A, B) = A_contrast_5_50 × B_contrast_5_50
  If both elements are locally elevated above the regional background,
  the product is positive and large → matches the deposit pattern.

fit():
  Learns which pairs genuinely co-enrich near deposits (AUC-weighted).
  Direction is also learned (+1: co-enrichment good; −1: one depleted is good).

Interpretability:
  knowledge_entry() returns the top pairs with their weights and pattern,
  readable by a geologist.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from core.data_source import DataSourceRegistry
from core.expert_tree import NodeScore

logger = logging.getLogger(__name__)

_SIGMOID_SCALE = 3.0   # co-enrichment values are small (product of contrasts); scale up


class ElementCorrelationExpert:
    """Scores by element pair co-enrichment in the local neighbourhood.

    Parameters
    ----------
    target      : target element symbol, e.g. "Cu"
    source_name : name of the AssaySource to read features from
    pairs       : list of (elem1, elem2) tuples to evaluate
    """

    _invert_score: bool = False

    def __init__(
        self,
        target: str,
        source_name: str,
        pairs: list[tuple[str, str]],
    ) -> None:
        self.name = f"{target.lower()}_{source_name}_correlation"
        self.required_sources: list[str] = [source_name]
        self._source_name = source_name
        self._pairs = pairs
        self._pair_weight:    dict[tuple[str, str], float] = {}
        self._pair_direction: dict[tuple[str, str], float] = {}
        self._fitted = False

    # ── fitting ───────────────────────────────────────────────────────────────

    def fit(self, positives: list, background: list) -> None:
        src = DataSourceRegistry.get(self._source_name)
        pos_rows = [fv for s in positives  if (fv := src.get_features(s.id)) is not None]
        bg_rows  = [fv for s in background if (fv := src.get_features(s.id)) is not None]
        if not pos_rows or not bg_rows:
            logger.warning("%s: insufficient data for fit", self.name)
            return

        for e1, e2 in self._pairs:
            k1 = f"{e1}_contrast_5_50"
            k2 = f"{e2}_contrast_5_50"

            def _coenrich(rows):
                vals = []
                for fv in rows:
                    v1, v2 = fv.get(k1, np.nan), fv.get(k2, np.nan)
                    if np.isfinite(v1) and np.isfinite(v2):
                        vals.append(v1 * v2)
                return np.array(vals)

            pos_co = _coenrich(pos_rows)
            bg_co  = _coenrich(bg_rows)

            if len(pos_co) < 5 or len(bg_co) < 5:
                self._pair_weight[(e1, e2)]    = 0.0
                self._pair_direction[(e1, e2)] = 1.0
                continue

            all_scores = np.concatenate([pos_co, bg_co])
            all_labels = np.array([1] * len(pos_co) + [0] * len(bg_co))
            try:
                auc = float(roc_auc_score(all_labels, all_scores))
            except Exception:
                auc = 0.5

            if auc < 0.5:
                direction = -1.0
                auc = 1.0 - auc
            else:
                direction = 1.0

            self._pair_weight[(e1, e2)]    = max(0.0, (auc - 0.5) * 2.0)
            self._pair_direction[(e1, e2)] = direction

        logger.info(
            "%s: fitted %d pairs, %d active (w>0)",
            self.name, len(self._pairs),
            sum(1 for w in self._pair_weight.values() if w > 0),
        )
        self._fitted = True

    def fit_kb(self, background: list) -> None:
        """Legacy path: equal weights, direction=+1."""
        for pair in self._pairs:
            self._pair_weight[pair]    = 1.0 / max(len(self._pairs), 1)
            self._pair_direction[pair] = 1.0
        self._fitted = True

    # ── scoring ───────────────────────────────────────────────────────────────

    def score_node(self, sample) -> Optional[NodeScore]:
        if not self._fitted:
            return None
        src = DataSourceRegistry.get(self._source_name)
        fv  = src.get_features(sample.id)
        if fv is None:
            return None

        total_s = total_w = 0.0
        z_scores: dict[str, float] = {}

        for (e1, e2), w in self._pair_weight.items():
            if w < 1e-9:
                continue
            k1, k2 = f"{e1}_contrast_5_50", f"{e2}_contrast_5_50"
            v1, v2 = fv.get(k1, np.nan), fv.get(k2, np.nan)
            if not (np.isfinite(v1) and np.isfinite(v2)):
                continue
            co = v1 * v2
            d  = self._pair_direction[(e1, e2)]
            feat_name = f"coenrich_{e1}_{e2}"
            z_scores[feat_name] = float(co * d)
            total_s += w * d * co
            total_w += w

        if total_w < 1e-9:
            return None

        raw = total_s / total_w
        if self._invert_score:
            raw = -raw
        sc = float(1.0 / (1.0 + np.exp(-raw * _SIGMOID_SCALE)))
        return NodeScore(
            name=self.name,
            score=sc,
            weights={f"coenrich_{e1}_{e2}": self._pair_weight[(e1, e2)]
                     for e1, e2 in self._pair_weight},
            feature_z=z_scores,
        )

    def score(self, sample) -> Optional[float]:
        ns = self.score_node(sample)
        return ns.score if ns is not None else None

    # ── interpretability ──────────────────────────────────────────────────────

    def knowledge_entry(self) -> dict:
        """Top co-enrichment pairs and their patterns — readable by a geologist."""
        ranked = sorted(
            [(w, e1, e2, self._pair_direction.get((e1, e2), 1.0))
             for (e1, e2), w in self._pair_weight.items() if w > 1e-9],
            reverse=True,
        )
        return {
            "expert": self.name,
            "top_pairs": [
                {
                    "pair": f"{e1}–{e2}",
                    "weight": round(w, 3),
                    "pattern": "co-enriched near deposit" if d > 0
                               else "one element depleted near deposit",
                }
                for w, e1, e2, d in ranked[:6]
            ],
        }
