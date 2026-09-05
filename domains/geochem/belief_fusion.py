"""BeliefFusionScorer — Dempster-Shafer fusion of expert-tree leaves.

Re-aggregates the leaf experts of a *fitted* CompositeNode as Dempster-Shafer
mass functions on a binary frame {P (prospective), B (barren)}, replacing the
default AUC-weighted linear average. Implemented as a post-processor over a
fitted tree — no change to the experts, data sources, or fitting code.

score -> mass mapping
---------------------
Each leaf expert produces a sigmoid score ``s`` in [0, 1] and a per-point
reliability ``r`` in [0, 1]::

    m({P})    = r * s
    m({B})    = r * (1 - s)
    m({P,B})  = 1 - r          # uncertainty: low reliability -> near-vacuous,
                               # i.e. the expert abstains at this point

reliability modes (the ``local_reliability`` design)
----------------------------------------------------
The whole point of a *local* reliability is to catch rare positives: a deposit
with an atypical signature is detected by only one expert, and an averaging
fusion washes it out. The fix is to make non-committal experts vacuous so they
become Dempster identity elements and stop diluting the lone real detector.

  "global"  r = global AUC weight (constant per expert) — baseline.
  "local"   r = per-expert local evidence strength at this point's score.
            A 1-D logistic recalibration label~score gives posterior(s);
            strength(s) = |posterior(s) - prior| / max_dev in [0,1].
            Non-discriminative experts -> slope~0 -> posterior~prior ->
            strength~0 -> vacuous everywhere. A rare, confident high score ->
            posterior >> prior -> strength~1 -> dominates the fusion.
  "hybrid"  r = sqrt(global_weight * strength(s)).

fusion
------
Leaf masses are combined with Dempster's rule. The fused mass yields
score = BetP({P}); bel = Bel({P}); pl = Pl({P}). The returned NodeScore keeps
the same tree shape (same leaf children, feature_z, weights) plus optional
bel / pl, so narrative.py is unaffected.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from core.belief import BeliefFunction, dempster_combine
from core.bk_fusion import outer_bk_revise
from core.expert_tree import CompositeNode, NodeScore

PROSPECTIVE = "P"
BARREN = "B"
FRAME = [PROSPECTIVE, BARREN]


def score_to_mass(
    score: float,
    reliability: float,
    frame: list[str] = FRAME,
) -> BeliefFunction:
    """Map a leaf (score, reliability) pair to a DS mass on {P, B}."""
    s = float(np.clip(score, 0.0, 1.0))
    r = float(np.clip(reliability, 0.0, 1.0))
    p, b = frame[0], frame[1]
    mass: dict = {}
    if r * s > 1e-12:
        mass[frozenset([p])] = r * s
    if r * (1.0 - s) > 1e-12:
        mass[frozenset([b])] = r * (1.0 - s)
    full = frozenset(frame)
    mass[full] = mass.get(full, 0.0) + (1.0 - r)
    return BeliefFunction(frame=list(frame), mass=mass)


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


class _ExpertCalib:
    """1-D logistic recalibration of one expert: score -> posterior(positive).

    evidence ``strength(s)`` measures how far the recalibrated posterior at
    score ``s`` departs from the base rate, normalised to [0, 1]. A flat
    (non-discriminative) expert has strength ~ 0 everywhere.
    """

    __slots__ = ("slope", "intercept", "prior", "max_dev")

    def __init__(self, slope: float, intercept: float, prior: float) -> None:
        self.slope = slope
        self.intercept = intercept
        self.prior = prior
        # posterior is monotone in s, so its extreme deviation from the prior
        # is at one of the endpoints of [0, 1].
        dev0 = abs(_sigmoid(intercept) - prior)
        dev1 = abs(_sigmoid(slope + intercept) - prior)
        self.max_dev = max(dev0, dev1)

    def strength(self, s: float) -> float:
        if self.max_dev < 1e-9:
            return 0.0
        post = _sigmoid(self.slope * float(s) + self.intercept)
        return float(np.clip(abs(post - self.prior) / self.max_dev, 0.0, 1.0))


def _child_signal(child, sample) -> Optional[tuple[NodeScore, float]]:
    """Score a child and return (NodeScore, score).

    Every node applies its own `_invert_score` inside score_node() (see
    CompositeNode), so ns.score is already the final, correctly-oriented value.
    Flipping it again here would cancel the leaf's flip and feed the
    anti-correlated signal into the fusion with a positive weight.
    """
    ns = child.score_node(sample)
    if ns is None:
        return None
    return ns, float(ns.score)


class BeliefFusionScorer:
    """Dempster-Shafer re-aggregation of a fitted expert tree's leaves.

    Parameters
    ----------
    tree : fitted CompositeNode whose leaves are re-aggregated.
    mode : "global" | "local" | "hybrid" — how per-leaf reliability is set.
    """

    def __init__(self, tree: CompositeNode, frame: list[str] = FRAME,
                 mode: str = "local") -> None:
        self._tree = tree
        self._frame = list(frame)
        self._mode = mode
        self._calib: dict[str, _ExpertCalib] = {}

    # ── fitting / calibration ───────────────────────────────────────────────
    def fit(self, positives: list, background: list) -> None:
        self._tree.fit(positives, background)
        self.calibrate(positives, background)

    def calibrate(self, positives: list, background: list) -> None:
        """Fit the per-expert score->posterior recalibration (modes local/hybrid)."""
        if self._mode == "global":
            return
        samples = list(positives) + list(background)
        labels0 = [1] * len(positives) + [0] * len(background)
        for child in self._tree.children:
            scores, labels = [], []
            for s, lbl in zip(samples, labels0):
                sig = _child_signal(child, s)
                if sig is not None:
                    scores.append(sig[1])
                    labels.append(lbl)
            if len(set(labels)) < 2 or len(scores) < 10:
                continue
            X = np.asarray(scores).reshape(-1, 1)
            try:
                lr = LogisticRegression(max_iter=200).fit(X, labels)
            except Exception:
                continue
            self._calib[child.name] = _ExpertCalib(
                slope=float(lr.coef_[0][0]),
                intercept=float(lr.intercept_[0]),
                prior=float(np.mean(labels)),
            )

    # ── per-leaf reliability components ─────────────────────────────────────
    def _global_r(self, child, sample) -> float:
        gw = self._tree._child_weights.get(child.name, 0.0)
        lr = getattr(child, "local_reliability", lambda _s: 1.0)(sample)
        return float(np.clip(gw * float(lr), 0.0, 1.0))

    def _strength(self, child, s: float) -> float:
        calib = self._calib.get(child.name)
        return float(np.clip(calib.strength(s), 0.0, 1.0)) if calib is not None else 0.0

    # ── scoring ─────────────────────────────────────────────────────────────
    def score_node(self, sample) -> Optional[NodeScore]:
        # pass 1: gather every leaf's signal (score, global reliability, strength)
        sigs = []  # (child, ns, s, r_global, strength)
        for child in self._tree.children:
            sig = _child_signal(child, sample)
            if sig is None:
                continue
            ns, s = sig
            sigs.append((child, ns, s, self._global_r(child, sample),
                         self._strength(child, s)))
        if not sigs:
            return None

        # routing weight: blend global->local by expert disagreement.
        # consensus (all experts agree) -> alpha 0 -> pure global (stable);
        # conflict (mixed votes) -> alpha 1 -> pure local (let the specific
        # expert dominate, catching the rare positive).
        alpha = 0.0
        if self._mode == "routed":
            pro = sum(rg * max(0.0, s - 0.5) for _, _, s, rg, _ in sigs)
            con = sum(rg * max(0.0, 0.5 - s) for _, _, s, rg, _ in sigs)
            denom = pro + con
            if denom > 1e-9:
                disagree = min(pro, con) / denom          # in [0, 0.5]
                alpha = float(min(1.0, 2.0 * disagree))

        # pass 2: set per-leaf reliability per mode, build masses
        child_scores: dict[str, NodeScore] = {}
        eff_weights: dict[str, float] = {}
        masses: list[BeliefFunction] = []
        for child, ns, s, rg, strength in sigs:
            if self._mode == "global":
                r = rg
            elif self._mode == "local":
                r = strength
            elif self._mode == "hybrid":
                r = math.sqrt(max(rg, 0.0) * strength)
            elif self._mode == "routed":
                r = (1.0 - alpha) * rg + alpha * strength
            else:
                r = rg
            r = float(np.clip(r, 0.0, 1.0))
            ns.confidence = r
            child_scores[child.name] = ns
            eff_weights[child.name] = r
            if r > 1e-9:
                masses.append(score_to_mass(s, r, self._frame))

        if not child_scores:
            return None

        p_hyp = frozenset([self._frame[0]])
        if masses:
            fused = dempster_combine(masses, self._frame)
            betp = fused.to_pignistic()
            score = float(betp.get(self._frame[0], 0.5))
            bel = float(fused.bel(p_hyp))
            pl = float(fused.pl(p_hyp))
        else:
            score = float(np.mean([c.score for c in child_scores.values()]))
            bel = pl = score

        return NodeScore(
            name=f"{self._tree.name}_belief",
            score=float(np.clip(score, 0.0, 1.0)),
            confidence=float(np.clip(np.mean(list(eff_weights.values())), 0.0, 1.0)),
            children=child_scores,
            weights=eff_weights,
            bel=bel,
            pl=pl,
        )

    def score(self, sample) -> Optional[float]:
        ns = self.score_node(sample)
        return ns.score if ns is not None else None


# ── Belief revision (Outer BK, Zhou 2014 Thm 3.5) ────────────────────────────

GEOCHEM_SUFFIXES = ("_enrichment", "_pathfinder", "_correlation")

# Fine frame: Mine-grade prospective / Deposit-grade prospective / Barren.
OMEGA = ["M", "D", "B"]
# Coarse frame the regional context evidence speaks to: it can only resolve
# prospective-vs-barren, not grade. Partition: PR = {M, D}, BA = {B}.
COARSEN = {
    frozenset(["PR"]): frozenset(["M", "D"]),
    frozenset(["BA"]): frozenset(["B"]),
}


def _is_geochem(name: str) -> bool:
    return name.endswith(GEOCHEM_SUFFIXES)


def _fuse_binary(pairs: list[tuple[float, float]],
                 frame: list[str] = FRAME) -> tuple[float, float]:
    """Dempster-fuse a group of (score, reliability) leaves on {P, B}.

    Returns (BetP(P), committed_mass) where committed_mass = 1 - vacuous mass.
    """
    masses = [score_to_mass(s, r, frame) for s, r in pairs if r > 1e-9]
    if not masses:
        return 0.5, 0.0
    fused = dempster_combine(masses, frame)
    s = float(fused.to_pignistic().get(frame[0], 0.5))
    committed = float(1.0 - fused.mass.get(frozenset(frame), 0.0))
    return s, committed


class BeliefRevisionScorer:
    """Outer Belief-Kinematics revision: geochem prior revised by context.

    Asymmetric fusion that uses the BK Jeffrey's rule (Zhou 2014, Thm 3.5,
    implemented in core.bk_fusion):

      * geochem experts -> a *fine* prior on Omega = {M, D, B}. A point-specific
        geochemical anomaly can distinguish grade (strong -> Mine-like,
        moderate -> Deposit-like, none -> Barren).
      * geophysics + structure experts -> *coarse* evidence on {PR, BA}. Regional
        context resolves only prospective-vs-barren, not grade.

    Outer BK reweights the {PR, BA} partition by the context evidence and
    redistributes mass within PR = {M, D} preserving the geochem M:D ratio.
    Prospectivity score = BetP(M) + BetP(D); bel/pl give the [Bel, Pl] interval
    for the prospective hypothesis {M, D}.

    This is the "basic version": the geochem->grade split is a monotone heuristic
    (m(M) ~ s^2, m(D) ~ s(1-s)); reliabilities use the global AUC weight.
    """

    def __init__(self, tree: CompositeNode) -> None:
        self._tree = tree

    def fit(self, positives: list, background: list) -> None:
        self._tree.fit(positives, background)

    def _global_r(self, child, sample) -> float:
        gw = self._tree._child_weights.get(child.name, 0.0)
        lr = getattr(child, "local_reliability", lambda _s: 1.0)(sample)
        return float(np.clip(gw * float(lr), 0.0, 1.0))

    def score_node(self, sample) -> Optional[NodeScore]:
        geo_pairs: list[tuple[float, float]] = []
        ctx_pairs: list[tuple[float, float]] = []
        children: dict[str, NodeScore] = {}

        for child in self._tree.children:
            sig = _child_signal(child, sample)
            if sig is None:
                continue
            ns, s = sig
            r = self._global_r(child, sample)
            ns.confidence = r
            children[child.name] = ns
            (geo_pairs if _is_geochem(child.name) else ctx_pairs).append((s, r))

        if not children:
            return None

        s_geo, r_geo = _fuse_binary(geo_pairs)
        s_ctx, r_ctx = _fuse_binary(ctx_pairs)

        # geochem fine prior on {M, D, B}
        pm: dict = {}
        if r_geo * s_geo * s_geo > 1e-12:
            pm[frozenset(["M"])] = r_geo * s_geo * s_geo
        if r_geo * s_geo * (1.0 - s_geo) > 1e-12:
            pm[frozenset(["D"])] = r_geo * s_geo * (1.0 - s_geo)
        if r_geo * (1.0 - s_geo) > 1e-12:
            pm[frozenset(["B"])] = r_geo * (1.0 - s_geo)
        omega_full = frozenset(OMEGA)
        pm[omega_full] = pm.get(omega_full, 0.0) + (1.0 - r_geo)
        prior = BeliefFunction(frame=list(OMEGA), mass=pm)

        # context coarse evidence on {PR, BA}
        em: dict = {}
        if r_ctx * s_ctx > 1e-12:
            em[frozenset(["PR"])] = r_ctx * s_ctx
        if r_ctx * (1.0 - s_ctx) > 1e-12:
            em[frozenset(["BA"])] = r_ctx * (1.0 - s_ctx)
        coarse_full = frozenset(["PR", "BA"])
        em[coarse_full] = em.get(coarse_full, 0.0) + (1.0 - r_ctx)
        evidence = BeliefFunction(frame=["PR", "BA"], mass=em)

        if r_ctx < 1e-9:
            revised = prior                       # no context → keep prior
        else:
            try:
                revised = outer_bk_revise(prior, evidence, COARSEN)
            except Exception:
                revised = prior

        betp = revised.to_pignistic()
        prosp = frozenset(["M", "D"])
        score = float(np.clip(betp.get("M", 0.0) + betp.get("D", 0.0), 0.0, 1.0))
        return NodeScore(
            name=f"{self._tree.name}_revision",
            score=score,
            confidence=float(np.clip(max(r_geo, r_ctx), 0.0, 1.0)),
            children=children,
            bel=float(revised.bel(prosp)),
            pl=float(revised.pl(prosp)),
        )

    def score(self, sample) -> Optional[float]:
        ns = self.score_node(sample)
        return ns.score if ns is not None else None
