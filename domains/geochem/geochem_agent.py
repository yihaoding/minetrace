"""Backward-compatibility shim.

Import from prospectivity_model instead:
    from domains.geochem.prospectivity_model import ProspectivityModel, TargetConfig, WA_BBOX
"""
from domains.geochem.prospectivity_model import (  # noqa: F401
    ProspectivityModel,
    ProspectivityModel as GeochemAgent,
    TargetConfig,
    TargetConfig as GeochemTargetConfig,
    PointEvidenceExpert,
    WA_BBOX,
)
