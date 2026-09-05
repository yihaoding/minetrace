"""Single-point semantic description — Stage 2.

Given a NodeScore tree produced by an expert tree for one geographic point,
generate a structured, human-readable narrative explaining the score.

Entry point:
    describe(node_score, target, x, y) -> PointNarrative

PointNarrative exposes:
    .to_text()      plain-text block (for terminal / logs)
    .to_markdown()  GitHub-flavoured markdown (for reports)
    .to_dict()      JSON-serialisable dict (for downstream use)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ── geological knowledge tables ───────────────────────────────────────────────

# Scale label → human name
_SCALE_LABEL: dict[str, str] = {
    "5km":  "5 km",
    "10km": "10 km",
    "50km": "50 km (regional)",
}

# Stat label → human name
_STAT_LABEL: dict[str, str] = {
    "median":     "median concentration",
    "p90":        "90th-percentile",
    "cv":         "coefficient of variation",
    "frac_above": "fraction above background threshold",
    "contrast_5_50":  "5 km vs 50 km contrast",
    "contrast_10_50": "10 km vs 50 km contrast",
}

# Pathfinder → geological significance per commodity system
_PATHFINDER_MEANING: dict[str, dict[str, str]] = {
    "Cu": {
        "Au": "Au–Cu co-enrichment (porphyry / IOCG indicator)",
        "Mo": "Mo enrichment (porphyry Cu–Mo indicator)",
        "Ag": "Ag co-enrichment (epithermal / IOCG indicator)",
        "Co": "Co anomaly (IOCG / mafic Cu system)",
        "Bi": "Bi enrichment (skarn / intrusion-related)",
    },
    "Au": {
        "Ag": "Ag–Au association (orogenic gold / epithermal)",
        "As": "As enrichment (orogenic gold pathfinder)",
        "Sb": "Sb anomaly (low-sulphidation epithermal indicator)",
        "Bi": "Bi enrichment (intrusion-related gold system)",
        "Mo": "Mo co-anomaly (porphyry-related gold)",
    },
    "W": {
        "Sn": "Sn–W co-enrichment (granite-related skarn / greisen)",
        "Mo": "Mo enrichment (W–Mo skarn / greisen)",
        "Bi": "Bi anomaly (intrusion-related W system)",
        "As": "As anomaly (distal skarn alteration)",
    },
    "Ni": {
        "Co": "Co–Ni association (komatiitic / ultramafic host)",
        "Cr": "Cr enrichment (ultramafic / peridotite indicator)",
        "Cu": "Cu–Ni co-enrichment (magmatic sulphide system)",
    },
}

# Expert type → short geological interpretation
_EXPERT_INTERP: dict[str, str] = {
    "enrichment":    "direct {target} enrichment in {source}",
    "pathfinder":    "pathfinder co-enrichment in {source}",
    "magnetic":      "magnetic anomaly (structural / lithological control)",
    "gravity":       "gravity anomaly (density contrast / intrusion)",
    "radiometric":   "radiometric signature (K–Th–U alteration halo)",
    "geochronology": "crustal-age boundary (terrane / structural control)",
}

# Score → tier
_TIERS = [
    (0.80, "Strong",    "score well above background; multiple independent lines of evidence"),
    (0.65, "Moderate",  "elevated above background; partial evidence convergence"),
    (0.52, "Weak",      "marginally elevated; low signal-to-noise"),
    (0.00, "Background","indistinguishable from random background"),
]


# ── data classes ─────────────────────────────────────────────────────────────

@dataclass
class FeatureSignal:
    """One feature's contribution at a single point."""
    feature: str        # raw feature name, e.g. "Au_5km_median"
    z_score: float      # signed z-score (direction-adjusted)
    weight: float       # shape weight (AUC-derived)
    expert_name: str    # which leaf expert produced this
    source: str         # data source name, e.g. "rockchips"

    @property
    def contribution(self) -> float:
        return self.weight * self.z_score

    def human_label(self) -> str:
        """Rewrite raw feature key into readable text."""
        parts = self.feature.split("_")
        # ratio features: "ratio_Cu_Mo"
        if parts[0] == "ratio":
            return f"{parts[1]}/{parts[2]} log-ratio"
        # PC features: "PC1_5km_median"
        if parts[0].startswith("PC"):
            pc = parts[0]
            scale = _SCALE_LABEL.get(parts[1], parts[1])
            return f"PCA {pc} at {scale}"
        # spatial coherence
        if "coherence" in self.feature:
            return "spatial clustering of top samples (10 km)"
        # element stat: "Au_5km_median" or "Au_contrast_5_50"
        elem = parts[0]
        if len(parts) >= 3 and parts[1] in _SCALE_LABEL:
            scale = _SCALE_LABEL[parts[1]]
            stat  = _STAT_LABEL.get(parts[2], parts[2])
            return f"{elem} {stat} ({scale})"
        if len(parts) >= 3 and parts[1] == "contrast":
            s1, s2 = parts[2], parts[3]
            return f"{elem} local/background contrast ({s1} km vs {s2} km)"
        return self.feature


@dataclass
class ExpertContrib:
    """One leaf expert's contribution summary."""
    name: str
    score: float
    tree_weight: float          # weight assigned by CompositeNode parent
    top_signals: list[FeatureSignal] = field(default_factory=list)

    @property
    def effective_contrib(self) -> float:
        return self.tree_weight * abs(self.score - 0.5)


@dataclass
class PointNarrative:
    """Complete semantic description of a single scored point."""
    x: float
    y: float
    target: str
    g_score: float
    tier: str
    tier_reason: str
    expert_contribs: list[ExpertContrib]   # sorted by effective_contrib desc
    top_signals: list[FeatureSignal]       # global top-N, sorted by |contribution|

    def to_text(self) -> str:
        lines: list[str] = []
        lat_str = f"{abs(self.y):.3f}°{'S' if self.y < 0 else 'N'}"
        lon_str = f"{self.x:.3f}°{'E' if self.x >= 0 else 'W'}"
        lines.append(f"Point: {lon_str}  {lat_str}")
        lines.append(f"Target: {self.target}  |  Score: {self.g_score:.3f}  [{self.tier}]")
        lines.append(f"Reason: {self.tier_reason}")
        lines.append("")

        if self.top_signals:
            lines.append("Top signals:")
            for sig in self.top_signals[:8]:
                arrow = "↑" if sig.z_score > 0 else "↓"
                lines.append(
                    f"  {arrow} {sig.human_label():<45s}"
                    f"  z={sig.z_score:+.1f}σ  w={sig.weight:.2f}"
                    f"  [{sig.source}]"
                )
            lines.append("")

        if self.expert_contribs:
            lines.append("Expert breakdown:")
            for ec in self.expert_contribs:
                bar = "█" * int(ec.score * 20)
                lines.append(
                    f"  {ec.name:<45s}"
                    f"  score={ec.score:.3f}  tree_w={ec.tree_weight:.2f}"
                )
            lines.append("")

        lines.append(self._geological_paragraph())
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lat_str = f"{abs(self.y):.3f}°{'S' if self.y < 0 else 'N'}"
        lon_str = f"{self.x:.3f}°{'E' if self.x >= 0 else 'W'}"

        tier_badge = {
            "Strong": "🟢", "Moderate": "🟡", "Weak": "🟠", "Background": "⚪"
        }.get(self.tier, "")

        lines: list[str] = [
            f"## {self.target} Prospectivity Report",
            f"**Location:** {lon_str} · {lat_str}  ",
            f"**Score:** `{self.g_score:.3f}` — {tier_badge} **{self.tier}**  ",
            f"**Summary:** {self.tier_reason}",
            "",
            "### Top geochemical signals",
            "| Signal | z-score | Weight | Source |",
            "|--------|---------|--------|--------|",
        ]
        for sig in self.top_signals[:8]:
            arrow = "↑" if sig.z_score > 0 else "↓"
            lines.append(
                f"| {arrow} {sig.human_label()} "
                f"| `{sig.z_score:+.1f}σ` "
                f"| {sig.weight:.2f} "
                f"| {sig.source} |"
            )

        lines += [
            "",
            "### Expert breakdown",
            "| Expert | Score | Tree weight |",
            "|--------|-------|-------------|",
        ]
        for ec in self.expert_contribs:
            lines.append(f"| {ec.name} | `{ec.score:.3f}` | {ec.tree_weight:.2f} |")

        lines += ["", "### Geological interpretation", self._geological_paragraph()]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "target": self.target,
            "g_score": self.g_score,
            "tier": self.tier,
            "tier_reason": self.tier_reason,
            "expert_contribs": [
                {
                    "name": ec.name,
                    "score": ec.score,
                    "tree_weight": ec.tree_weight,
                    "top_signals": [
                        {"feature": s.feature, "z_score": s.z_score,
                         "weight": s.weight, "source": s.source}
                        for s in ec.top_signals[:5]
                    ],
                }
                for ec in self.expert_contribs
            ],
            "top_signals": [
                {"feature": s.feature, "human_label": s.human_label(),
                 "z_score": s.z_score, "weight": s.weight,
                 "expert": s.expert_name, "source": s.source}
                for s in self.top_signals[:10]
            ],
        }

    # ── private ───────────────────────────────────────────────────────────────

    def _geological_paragraph(self) -> str:
        target = self.target
        if self.tier == "Background":
            return (
                f"This location shows no significant {target} anomaly relative to "
                f"the regional background. No expert provided meaningful signal."
            )

        active = [ec for ec in self.expert_contribs if ec.tree_weight > 0.05 and ec.score > 0.52]
        if not active:
            return (
                f"Score is {self.g_score:.2f} ({self.tier.lower()}), but no single expert "
                f"dominates. Consider running ablation to identify the primary driver."
            )

        # Lead sentence from top expert
        top_ec = active[0]
        etype = _expert_type(top_ec.name)
        source = _expert_source(top_ec.name)
        interp = _EXPERT_INTERP.get(etype, etype).format(target=target, source=source)
        para = f"The score of {self.g_score:.3f} ({self.tier.lower()}) is primarily driven by {interp}."

        # pathfinder clues
        pf_signals = [s for s in self.top_signals if _is_pathfinder(s.feature, target)]
        if pf_signals:
            pf_names = _unique_elements(pf_signals[:3])
            pf_meaning = [
                _PATHFINDER_MEANING.get(target, {}).get(e, f"{e} co-enrichment")
                for e in pf_names
            ]
            para += (
                f" Key pathfinder evidence: {'; '.join(pf_meaning[:2])}."
            )

        # scale observation
        local_strong = any(
            "5km" in s.feature or "10km" in s.feature
            for s in self.top_signals[:4] if s.z_score > 2
        )
        regional_strong = any(
            "50km" in s.feature
            for s in self.top_signals[:4] if s.z_score > 2
        )
        if local_strong and not regional_strong:
            para += " Signal is near-field (5–10 km scale), suggesting proximity to a discrete source."
        elif regional_strong and not local_strong:
            para += " Signal is regional (50 km scale), consistent with a distal or dispersed mineralised system."
        elif local_strong and regional_strong:
            para += " Signal is present at both local and regional scales, supporting a significant mineralised system."

        # geophysics note if relevant
        geo_experts = [ec for ec in active if _expert_type(ec.name) in
                       ("magnetic", "gravity", "radiometric", "geochronology")]
        if geo_experts:
            geo_names = ", ".join(_EXPERT_INTERP.get(_expert_type(e.name), e.name).split("(")[0].strip()
                                  for e in geo_experts[:2])
            para += f" Geophysical support: {geo_names}."

        return para


# ── helpers ───────────────────────────────────────────────────────────────────

def _expert_type(name: str) -> str:
    """Extract type token from expert name, e.g. 'cu_rockchips_enrichment' → 'enrichment'."""
    for token in ("enrichment", "pathfinder", "magnetic", "gravity",
                  "radiometric", "geochronology", "geology", "fault", "worm"):
        if token in name:
            return token
    return name.split("_")[-1]


def _split_key(key: str) -> tuple[str | None, str]:
    """Split a feature_z key into (source, bare_feature).

    Multi-source leaves emit "sediment:Cu_contrast_5_50"; single-source leaves
    emit the bare feature name. The bare name is what human_label() and the
    pathfinder matcher parse, so the prefix must be stripped before display.
    """
    if ":" in key:
        src, feat = key.split(":", 1)
        return src, feat
    return None, key


def _expert_source(name: str, feature: str = "") -> str:
    """Data source for a signal: the feature key's "src:" prefix if it has one,
    else inferred from the expert name ('cu_rockchips_enrichment' → 'rockchips').

    Multi-source experts are named 'cu_enrichment' — no source token — so the
    key prefix is the only place the true source is recorded.
    """
    src, _ = _split_key(feature)
    if src:
        return src
    for token in (
        "rockchips", "sediment", "drillhole", "shallowdrill", "soil",
        "geophysics", "geology", "structure", "magnetic", "gravity",
        "radiometric", "geochron", "fault", "worm",
    ):
        if token in name:
            return token
    return "unknown"


def _is_pathfinder(feature: str, target: str) -> bool:
    """True if feature belongs to a pathfinder element for this target."""
    pfinders = set(_PATHFINDER_MEANING.get(target, {}).keys())
    parts = feature.split("_")
    return len(parts) >= 1 and parts[0] in pfinders


def _unique_elements(signals: list[FeatureSignal]) -> list[str]:
    """Extract unique element names from a list of feature signals."""
    seen: list[str] = []
    for s in signals:
        parts = s.feature.split("_")
        if parts and parts[0] not in seen and not parts[0].startswith("PC") and parts[0] != "ratio":
            seen.append(parts[0])
    return seen


def _get_tier(score: float) -> tuple[str, str]:
    for threshold, tier, reason in _TIERS:
        if score >= threshold:
            return tier, reason
    return "Background", _TIERS[-1][2]


# ── main entry point ──────────────────────────────────────────────────────────

def describe(
    node_score,
    target: str,
    x: float,
    y: float,
    top_n: int = 10,
) -> PointNarrative:
    """Build a PointNarrative from a NodeScore tree.

    Parameters
    ----------
    node_score : NodeScore  (root of the expert tree output for this point)
    target     : commodity string, e.g. "Cu"
    x, y       : geographic coordinates (lon, lat)
    top_n      : how many top signals to include

    Returns
    -------
    PointNarrative
    """
    g = node_score.score
    tier, tier_reason = _get_tier(g)

    # ── collect per-expert leaf contributions ─────────────────────────────
    # top-level children of the root CompositeNode are the leaf (or sub-composite) experts
    root_weights = node_score.weights  # {child_name: tree_weight}

    expert_contribs: list[ExpertContrib] = []
    for child_name, child_ns in node_score.children.items():
        tw = root_weights.get(child_name, 0.0)
        # Collect feature signals from this child (leaf or nested composite).
        # z and weight must come from the SAME node: multi-source leaves key
        # feature_z by "src:feat" while their `weights` is keyed by source name,
        # so looking a feature key up in child_ns.weights silently returns 0.
        feat_signals = [
            FeatureSignal(
                feature=_split_key(key)[1],
                z_score=z,
                weight=w,
                expert_name=expert_name,
                source=_expert_source(expert_name, key),
            )
            for key, (z, w, expert_name) in child_ns.all_feature_signals().items()
            if abs(z) > 0.1
        ]
        feat_signals.sort(key=lambda s: abs(s.contribution), reverse=True)
        expert_contribs.append(
            ExpertContrib(
                name=child_name,
                score=child_ns.score,
                tree_weight=tw,
                top_signals=feat_signals[:5],
            )
        )

    expert_contribs.sort(key=lambda ec: ec.effective_contrib, reverse=True)

    # ── global top signals across all leaves ─────────────────────────────
    # weight comes from the owning leaf's feature_w, keyed identically to
    # feature_z, so contribution = weight × z is never silently zeroed.
    all_signals = [
        FeatureSignal(
            feature=_split_key(key)[1],
            z_score=z,
            weight=w,
            expert_name=expert_name,
            source=_expert_source(expert_name, key),
        )
        for key, (z, w, expert_name) in node_score.all_feature_signals().items()
    ]
    all_signals.sort(key=lambda s: abs(s.contribution), reverse=True)

    # deduplicate by (source, feature) — the same feature measured by two
    # different assay sources is two independent pieces of evidence
    seen_feats: set[tuple[str, str]] = set()
    top_signals: list[FeatureSignal] = []
    for sig in all_signals:
        k = (sig.source, sig.feature)
        if k not in seen_feats:
            seen_feats.add(k)
            top_signals.append(sig)
        if len(top_signals) >= top_n:
            break

    return PointNarrative(
        x=x,
        y=y,
        target=target,
        g_score=g,
        tier=tier,
        tier_reason=tier_reason,
        expert_contribs=expert_contribs,
        top_signals=top_signals,
    )
