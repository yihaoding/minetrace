"""Outer Belief-Kinematics (BK) fusion — Zhou 2014, UAI.

Reference: Zhou 2014, "Belief-Kinematics Jeffrey's Rules in the Theory of
Evidence", UAI. Core result: Theorem 3.5 (Outer Revision).

This file is FULLY domain-agnostic: no geochem, medical, or timeseries terms.
It operates on BeliefFunction objects only.

Key notation (matching paper):
  Ω  = fine frame (prior belief lives here)
  B  = coarse frame (evidence belief lives here, a partition of Ω)
  coarsening: maps each element of B → subset of Ω
              (many-to-one projection from Ω to B)
"""
from __future__ import annotations

import math

from core.belief import BeliefFunction


# ---------------------------------------------------------------------------
# Outer BK Revision — Theorem 3.5
# ---------------------------------------------------------------------------

def outer_bk_revise(
    bel_prior: BeliefFunction,
    bel_evidence: BeliefFunction,
    coarsening: dict[frozenset[str], frozenset[str]],
) -> BeliefFunction:
    """Revise bel_prior (on Ω) with bel_evidence (on coarse B) via Outer BK.

    coarsening maps each singleton B-element → its preimage in Ω.
    Non-singleton focal elements of bel_evidence are handled via set union.

    Returns a new BeliefFunction on Ω.

    Zhou 2014, Theorem 3.5:
      bel'(A) = Σ_{B ∈ focal(bel_e)}  bel^out_B(A | B) · q_B
    where
      q_B = Σ_{E: π(E)=B} m_prior(E)   (mass redistribution weights)
      bel^out_B(A|B) uses the outer conditioning: m/B(C) = Σ_{E⊆B} m(C∪E)
    """
    omega = frozenset(bel_prior.frame)

    # --- Step 1: compute coarsened mass q_B for each B-element -----------
    # For each element b ∈ frame_B, its preimage in Ω is coarsening[{b}].
    # q_B = Σ_{focal A of prior: coarsen(A) ⊆ B} m_prior(A)
    # We represent B-elements as frozensets of B-singletons.
    # Here we work directly: for a focal element E ⊆ Ω, its coarsened
    # image is the B-element that contains all of π(ω) for ω ∈ E.

    # Build reverse map: each Ω-singleton → its B-singleton
    omega_to_b: dict[str, str] = {}
    for b_elem, omega_preimage in coarsening.items():
        assert len(b_elem) == 1, "coarsening keys must be B-singletons"
        b_singleton = next(iter(b_elem))
        for omega_atom in omega_preimage:
            omega_to_b[omega_atom] = b_singleton

    def coarsen_set(omega_subset: frozenset[str]) -> frozenset[str]:
        """Map a subset of Ω to its image in B."""
        return frozenset(omega_to_b[w] for w in omega_subset if w in omega_to_b)

    # q_B: for each B-element, total prior mass whose coarsened image ⊆ B
    q: dict[frozenset[str], float] = {}
    for A, m in bel_prior.mass.items():
        if not A or m <= 0:
            continue
        b_image = coarsen_set(A)
        q[b_image] = q.get(b_image, 0.0) + m

    # --- Step 2: for each focal element of bel_evidence ------------------
    # The evidence provides updated weights p_B for each B-focal-element B.
    # We need bel^out_B(· | B) for the outer conditioning.

    new_mass: dict[frozenset[str], float] = {}

    for B_focal, p_B in bel_evidence.mass.items():
        if not B_focal or p_B <= 0:
            continue

        # Preimage of B_focal in Ω
        b_preimage: frozenset[str] = frozenset()
        for b_atom in B_focal:
            b_preimage = b_preimage | coarsening.get(frozenset([b_atom]), frozenset())

        # Outer conditioning of bel_prior on B_focal (via b_preimage in Ω)
        # m/B(C) = Σ_{E ⊆ Ω\b_preimage} m_prior(C ∪ E)   for C ⊆ b_preimage
        # This yields a new BeliefFunction on b_preimage.
        outer_cond = _outer_condition(bel_prior, b_preimage, omega)

        # Merge outer_cond mass into new_mass, weighted by p_B
        for C, m_C in outer_cond.items():
            if m_C > 0:
                new_mass[C] = new_mass.get(C, 0.0) + p_B * m_C

    if not new_mass:
        # Fall back to vacuous if nothing fired
        new_mass = {omega: 1.0}

    # Normalise floating-point drift
    total = sum(new_mass.values())
    new_mass = {k: v / total for k, v in new_mass.items()}

    return BeliefFunction(frame=bel_prior.frame, mass=new_mass)


def _outer_condition(
    bel: BeliefFunction,
    b_preimage: frozenset[str],
    omega: frozenset[str],
) -> dict[frozenset[str], float]:
    r"""Compute the outer-conditioned mass m/B for focal elements ⊆ b_preimage.

    m/B(C) = Σ_{E ⊆ (Ω\B)}  m(C ∪ E)   for C ⊆ B
    (Zhou 2014, definition before Theorem 3.5)
    """
    outside = omega - b_preimage
    result: dict[frozenset[str], float] = {}

    for A, m in bel.mass.items():
        if not A or m <= 0:
            continue
        # Split A into part inside B and part outside B
        inside_part = A & b_preimage
        outside_part = A - b_preimage
        # outside_part must be ⊆ (Ω\B); inside_part goes to C
        # (if outside_part ⊄ Ω\B this focal element is inconsistent — skip)
        if not outside_part.issubset(outside):
            continue
        C = inside_part  # C ⊆ B
        result[C] = result.get(C, 0.0) + m

    # Normalise: outer conditioning normalises over focal elements with C ≠ ∅
    total = sum(v for k, v in result.items() if k)
    if total < 1e-12:
        # Degenerate: put all mass on b_preimage
        return {b_preimage: 1.0}
    return {k: v / total for k, v in result.items() if k}


# ---------------------------------------------------------------------------
# Distance measure D_out — Zhou 2014, Section 4
# ---------------------------------------------------------------------------

def d_out(
    bel1: BeliefFunction,
    bel2: BeliefFunction,
    coarsening: dict[frozenset[str], frozenset[str]],
) -> float:
    """D_out distance between two belief functions on Ω under coarsening.

    Measures how differently bel1 and bel2 react to the same coarse evidence.
    Higher D_out → the two beliefs differ more in the coarsened space.
    """
    omega = frozenset(bel1.frame)
    assert frozenset(bel2.frame) == omega, "Both belief functions must share the same frame"

    omega_to_b: dict[str, str] = {}
    for b_elem, preimage in coarsening.items():
        b_singleton = next(iter(b_elem))
        for w in preimage:
            omega_to_b[w] = b_singleton

    def coarsen_set(s: frozenset[str]) -> frozenset[str]:
        return frozenset(omega_to_b[w] for w in s if w in omega_to_b)

    # Collect all B-focal-elements seen in either belief
    all_b_focals: set[frozenset[str]] = set()
    for A in bel1.mass:
        if A:
            all_b_focals.add(coarsen_set(A))
    for A in bel2.mass:
        if A:
            all_b_focals.add(coarsen_set(A))

    dist = 0.0
    for B_f in all_b_focals:
        # Build a dummy bel_e that is certain on B_f
        b_frame = list({w for bf in all_b_focals for w in bf})
        bel_e = BeliefFunction(frame=b_frame, mass={B_f: 1.0})

        bel1_rev = outer_bk_revise(bel1, bel_e, coarsening)
        bel2_rev = outer_bk_revise(bel2, bel_e, coarsening)

        # Jousselme distance on revised beliefs (simplified: L2 on pignistic)
        p1 = bel1_rev.to_pignistic()
        p2 = bel2_rev.to_pignistic()
        atoms = set(p1) | set(p2)
        sq = sum((p1.get(a, 0) - p2.get(a, 0)) ** 2 for a in atoms)
        dist += math.sqrt(sq)

    return dist / max(len(all_b_focals), 1)
