"""Test set construction and evaluation utilities for Phase 2+.

Test set design:
  - 100 positive samples: randomly selected from Cu Mine/Deposit sites
  - 100 negative samples: randomly generated geographic coordinates within
    WA bounding box, guaranteed to have at least MIN_COVERAGE sediment
    samples within 10 km (so features can be computed).

Training / KB set: remaining Cu Mine/Deposit sites not in test set.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import roc_auc_score

from domains.geochem.samples import GeochemSample, GeochemInstance, is_high_confidence_cu

logger = logging.getLogger(__name__)

# WA bounding box defaults (conservative, avoids ocean edges).
# Pass bbox=(x_min, x_max, y_min, y_max) to build_test_set for other regions.
WA_X_MIN, WA_X_MAX = 114.5, 128.5
WA_Y_MIN, WA_Y_MAX = -33.5, -15.0
WA_BBOX = (WA_X_MIN, WA_X_MAX, WA_Y_MIN, WA_Y_MAX)

N_POS_TEST = 100
N_NEG_TEST = 100
MIN_COVERAGE = 5   # sediment samples within 10 km required for valid background point


def build_test_set(
    catalog_instances: list[GeochemInstance],
    sediment_source,
    n_pos: int = N_POS_TEST,
    n_neg: int = N_NEG_TEST,
    seed: int = 42,
    bbox: tuple[float, float, float, float] = WA_BBOX,
    min_coverage: int = MIN_COVERAGE,
) -> tuple[list[GeochemSample], list[int], list[GeochemInstance]]:
    """Build evaluation set and training instances.

    Returns
    -------
    test_samples  : list of GeochemSample (positives + random negatives)
    test_labels   : list of int (1 = positive, 0 = negative)
    train_instances : GeochemInstance list for KB fitting (positives not in test)
    """
    rng = np.random.default_rng(seed)

    # ── positive test samples ──────────────────────────────────────────────
    all_pos = list(catalog_instances)
    if len(all_pos) < n_pos:
        raise ValueError(f"Need {n_pos} positives, only {len(all_pos)} available.")
    test_pos_idx = rng.choice(len(all_pos), size=n_pos, replace=False)
    test_pos_set = {all_pos[i].sample.id for i in test_pos_idx}
    test_pos_samples = [all_pos[i].sample for i in test_pos_idx]
    train_instances = [inst for inst in all_pos if inst.sample.id not in test_pos_set]

    logger.info("Test positives: %d | Train instances: %d", len(test_pos_samples), len(train_instances))

    # ── negative test samples (random geographic coordinates) ─────────────
    test_neg_samples: list[GeochemSample] = []
    attempts = 0
    max_attempts = n_neg * 200

    x_min, x_max, y_min, y_max = bbox
    while len(test_neg_samples) < n_neg and attempts < max_attempts:
        attempts += 1
        x = float(rng.uniform(x_min, x_max))
        y = float(rng.uniform(y_min, y_max))
        # Check sediment coverage at 10 km
        n_cover = sediment_source.compute_for_point(x, y).get("n_samples_10km", 0)
        if n_cover < min_coverage:
            continue
        sid = f"BG_{len(test_neg_samples):04d}"
        s = GeochemSample(site_code=sid, x=x, y=y)
        test_neg_samples.append(s)

    if len(test_neg_samples) < n_neg:
        logger.warning(
            "Only %d/%d background points found with sufficient coverage after %d attempts.",
            len(test_neg_samples), n_neg, attempts,
        )

    logger.info("Test negatives: %d random background points", len(test_neg_samples))

    # Register background points in sediment source
    sediment_source.register_samples(test_neg_samples)

    test_samples = test_pos_samples + test_neg_samples
    test_labels = [1] * len(test_pos_samples) + [0] * len(test_neg_samples)

    return test_samples, test_labels, train_instances


# ── evaluation metrics ────────────────────────────────────────────────────────

def compute_auc(scored_samples, labels) -> float:
    y_score = [r.g_score for r in scored_samples]
    return float(roc_auc_score(labels, y_score))


def expert_auc(scored_samples, labels, expert_name: str) -> float:
    y_true, y_score = [], []
    for r, label in zip(scored_samples, labels):
        if expert_name in r.expert_scores:
            y_true.append(label)
            y_score.append(r.expert_scores[expert_name])
    if len(y_true) == 0 or sum(y_true) == 0:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def top_k_precision(scored_samples, labels, k: int) -> float:
    paired = sorted(zip(scored_samples, labels), key=lambda t: t[0].g_score, reverse=True)
    top = paired[:k]
    hits = sum(label for _, label in top)
    return hits / k
