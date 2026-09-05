"""Structure / geology experts — three leaf nodes for the StructureNode.

FaultExpert    : fault distance + density (closer / denser → anomalous)
WormExpert     : magnetic & gravity worm distances
GeologyExpert  : one-hot rock type + age + craton + Cenozoic cover
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from core.data_source import DataSourceRegistry
from core.expert_tree import NodeScore
from domains.geochem.experts.geophysics_experts import _GeophysicsExpert as _ShapeLeafExpert

logger = logging.getLogger(__name__)


class FaultExpert(_ShapeLeafExpert):
    """Proximity to fault / shear zones and local fault density.

    Note: distance features are inverted (closer = more anomalous).
    The shape score handles direction automatically via per-feature AUC.
    """

    def __init__(self, source_name: str = "structure") -> None:
        super().__init__(
            name=f"fault_{source_name}",
            source_name=source_name,
            feature_keys=["dist_fault_km", "fault_density_5km", "fault_density_10km"],
        )


class WormExpert(_ShapeLeafExpert):
    """Proximity to geophysical gradient worms (mag + grav)."""

    def __init__(self, source_name: str = "structure") -> None:
        super().__init__(
            name=f"worm_{source_name}",
            source_name=source_name,
            feature_keys=["dist_worm_mag_km", "dist_worm_grav_km"],
        )


class GeologyExpert(_ShapeLeafExpert):
    """Rock-type one-hot encoding, age, craton membership, and cover flag.

    One-hot features are binary (0/1); shape score will select which rock
    types are actually enriched in the training set for each metal.
    """

    def __init__(self, source_name: str = "structure") -> None:
        super().__init__(
            name=f"geology_{source_name}",
            source_name=source_name,
            feature_keys=[
                "is_rt_granitic", "is_rt_felsic", "is_rt_mafic",
                "is_rt_ultramafic", "is_rt_metased", "is_rt_sedimentary",
                "is_rt_metamorphic", "is_rt_hydrothermal",
                "age_ma", "is_yilgarn", "is_pilbara", "is_covered",
            ],
        )
