"""Compatibility module forwarding to the defence package.

Ensures both `import defense` and `import defence` work identically.
"""
from defence.base import (
    BaseDefense,
    DefensePipeline,
    ScanResult,
    SpanSuspicion,
    get_defense,
)
from defence.layer1 import Layer1Defense
from defence.layer2 import Layer2Defense, ProvenanceRecord
from defence.layer3 import CausalProbeResult, Layer3Defense
from defence.layer4 import Layer4Defense
from defence.layer5 import CrossSourceProbeResult, Layer5Defense

__all__ = [
    "BaseDefense",
    "DefensePipeline",
    "ScanResult",
    "SpanSuspicion",
    "Layer1Defense",
    "Layer2Defense",
    "Layer3Defense",
    "Layer4Defense",
    "Layer5Defense",
    "ProvenanceRecord",
    "CausalProbeResult",
    "CrossSourceProbeResult",
    "get_defense",
]

