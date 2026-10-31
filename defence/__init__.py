"""Prompt Injection Defense (PID) package.

Modular layered defense architecture:
  - Layer 1 (defence.layer1): Tag and scan (reproduced, cheap)
  - Layer 2 (defence.layer2): Provenance / grounding check (reproduced)
  - Layer 3 (defence.layer3): Single-span causal attribution (leave-one-out)
  - Layer 4 (defence.layer4): Windowed / paired-span removal with pruning
  - Layer 5 (defence.layer5): Cross-source removal (planned stretch)
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
