"""Base architecture for Prompt Injection Defense (PID).

Defines the core interfaces, suspicion scoring dataclasses, span models,
and pipeline aggregation used across all defense layers.
"""
from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Suspicion & Span Models ───────────────────────────────────────────────────

@dataclass
class SpanSuspicion:
    """Represents a scored text span analyzed by pattern or classifier scanners.

    Attributes:
        span_text: The extracted substring (sentence, clause, or line).
        start_idx: Character start offset within the original tool output.
        end_idx: Character end offset within the original tool output.
        score: Suspicion score from 0.0 (benign) to 1.0 (highly suspicious).
        matched_patterns: List of pattern category names triggered in this span.
        metadata: Optional dictionary with rule details, regex names, etc.
    """
    span_text: str
    start_idx: int
    end_idx: int
    score: float
    matched_patterns: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanResult:
    """Result of scanning an untrusted content block.

    Reused by downstream layers (e.g. Layer 4 for pruning or Layer 3 for policy gating).

    Attributes:
        spans: Individual analyzed spans and their suspicion scores.
        overall_score: Peak or aggregated suspicion score (0.0 to 1.0).
        is_suspicious: Whether overall_score meets or exceeds threshold.
        tagged_content: Delimited and formatted content ready for agent context.
        metadata: Summary metadata including tool name, scan timestamp, etc.
    """
    spans: list[SpanSuspicion] = field(default_factory=list)
    overall_score: float = 0.0
    is_suspicious: bool = False
    tagged_content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def prune_spans(self, threshold: float = 0.5, replacement: str = "[PRUNED_SUSPICIOUS_CONTENT]") -> str:
        """Utility for Layer 4: Prunes spans exceeding the suspicion threshold.

        Returns cleaned text with suspicious instructions removed.
        """
        raw_text = self.metadata.get("raw_content", "")
        if not raw_text or not self.spans:
            return raw_text

        # Sort spans by start index
        suspicious_spans = sorted(
            [s for s in self.spans if s.score >= threshold],
            key=lambda s: s.start_idx
        )
        if not suspicious_spans:
            return raw_text

        # Reconstruct text replacing flagged ranges
        pieces = []
        cursor = 0
        for s in suspicious_spans:
            if s.start_idx > cursor:
                pieces.append(raw_text[cursor:s.start_idx])
            pieces.append(f"{replacement} (reason: {', '.join(s.matched_patterns)})")
            cursor = max(cursor, s.end_idx)

        if cursor < len(raw_text):
            pieces.append(raw_text[cursor:])

        return "".join(pieces)


# ── Base Defense Interface ────────────────────────────────────────────────────

class BaseDefense(ABC):  # noqa: B024 — hooks have safe default implementations
    """Abstract base class for all defense layers."""

    name: str = "base_defense"
    layer: int = 0
    description: str = "Base defense class"

    def __init__(self):
        self.last_scan_result: ScanResult | None = None

    def protect_system_prompt(self, system_prompt: str) -> str:
        """Transform or harden the system prompt."""
        return system_prompt

    def protect_user_prompt(self, user_prompt: str) -> str:
        """Preprocess or protect the incoming user prompt."""
        return user_prompt

    def protect_tool_result(
        self,
        tool_name: str,
        result: str,
        tool_args: dict[str, Any] | None = None,
        user_task: str = "",
    ) -> str:
        """Sanitize, isolate, or wrap content returned by a tool."""
        return result

    def validate_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> tuple[bool, str | None]:
        """Validate whether a tool call should be allowed to execute.

        Returns:
            (is_allowed, optional_block_reason)
        """
        return True, None

    def on_turn_start(
        self,
        turn: int,
        messages: list[dict[str, Any]],
        user_task: str,
    ) -> list[dict[str, Any]]:
        """Optional hook invoked before each LLM turn."""
        return messages


# ── Defense Pipeline ──────────────────────────────────────────────────────────

class DefensePipeline(BaseDefense):
    """Composite pipeline that executes multiple defense layers in sequence."""

    def __init__(self, layers: list[BaseDefense] | None = None):
        super().__init__()
        self.layers: list[BaseDefense] = layers or []

    @property
    def name(self) -> str:
        if not self.layers:
            return "none"
        return "+".join(layer.name for layer in self.layers)

    def add_layer(self, layer: BaseDefense) -> DefensePipeline:
        self.layers.append(layer)
        return self

    def get_layer(self, layer_number: int) -> BaseDefense | None:
        """Retrieve a defense layer by its layer integer."""
        for layer in self.layers:
            if getattr(layer, "layer", None) == layer_number:
                return layer
        return None

    def protect_system_prompt(self, system_prompt: str) -> str:
        for layer in self.layers:
            system_prompt = layer.protect_system_prompt(system_prompt)
        return system_prompt

    def protect_user_prompt(self, user_prompt: str) -> str:
        for layer in self.layers:
            user_prompt = layer.protect_user_prompt(user_prompt)
        return user_prompt

    def protect_tool_result(
        self,
        tool_name: str,
        result: str,
        tool_args: dict[str, Any] | None = None,
        user_task: str = "",
    ) -> str:
        for layer in self.layers:
            result = layer.protect_tool_result(tool_name, result, tool_args, user_task)
            if hasattr(layer, "last_scan_result") and layer.last_scan_result:
                self.last_scan_result = layer.last_scan_result
        return result

    def validate_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> tuple[bool, str | None]:
        for layer in self.layers:
            allowed, reason = layer.validate_tool_call(tool_name, tool_args, context)
            if not allowed:
                return False, reason
        return True, None

    def on_turn_start(
        self,
        turn: int,
        messages: list[dict[str, Any]],
        user_task: str,
    ) -> list[dict[str, Any]]:
        for layer in self.layers:
            messages = layer.on_turn_start(turn, messages, user_task)
        return messages


# ── Factory Helpers ───────────────────────────────────────────────────────────

def get_defense(spec: str | int | BaseDefense | None) -> BaseDefense | None:
    """Resolve a defense specification into a BaseDefense instance.

    Supported values:
      - None, 0, 'none', 'undefended'           -> None (Phase 0 baseline)
      - 1, '1', 'layer1', 'layer_1', 'l1'       -> Layer1Defense() (Tag and scan)
      - 2, '2', 'layer2', 'layer_2', 'l2'       -> Layer2Defense() (Provenance / grounding check)
      - 3, '3', 'layer3', 'layer_3', 'l3'       -> Layer3Defense() (Single-span causal attribution)
      - 4, '4', 'layer4', 'layer_4', 'l4'       -> Layer4Defense() (Windowed / paired-span removal & pruning)
      - '1+2', 'layer1+2', 'c1'                 -> DefensePipeline([Layer1, Layer2])
      - '1+2+3', 'layer1+2+3'                   -> DefensePipeline([Layer1, Layer2, Layer3])
      - '4layer', '1+2+3+4', 'all', 'full'      -> DefensePipeline([Layer1, Layer2, Layer3, Layer4]) (All 4 layers)
      - BaseDefense instance                    -> returned as-is
    """
    if spec is None:
        return None

    if isinstance(spec, BaseDefense):
        return spec

    spec_str = str(spec).strip().lower()

    if spec_str in ("0", "none", "undefended", "disabled", "false"):
        return None

    if spec_str in ("1", "layer1", "layer_1", "l1"):
        from defence.layer1 import Layer1Defense
        return Layer1Defense()

    if spec_str in ("2", "layer2", "layer_2", "l2"):
        from defence.layer2 import Layer2Defense
        return Layer2Defense()

    if spec_str in ("3", "layer3", "layer_3", "l3"):
        from defence.layer3 import Layer3Defense
        return Layer3Defense()

    if spec_str in ("4", "layer4", "layer_4", "l4"):
        from defence.layer4 import Layer4Defense
        return Layer4Defense()

    if spec_str in ("5", "layer5", "layer_5", "l5"):
        from defence.layer5 import Layer5Defense
        return Layer5Defense()

    if spec_str in ("1+2", "layer1+2", "layer1_2", "c1", "base_defense", "pipeline12"):
        from defence.layer1 import Layer1Defense
        from defence.layer2 import Layer2Defense
        return DefensePipeline([Layer1Defense(), Layer2Defense()])

    if spec_str in ("1+2+3", "layer1+2+3", "layer1_2_3", "c2", "pipeline123"):
        from defence.layer1 import Layer1Defense
        from defence.layer2 import Layer2Defense
        from defence.layer3 import Layer3Defense
        return DefensePipeline([Layer1Defense(), Layer2Defense(), Layer3Defense()])

    if spec_str in (
        "1+2+3+4",
        "layer1+2+3+4",
        "layer1_2_3_4",
        "4layer",
        "4-layer",
        "4layers",
        "four_layer",
        "c3",
        "pipeline1234",
    ):
        from defence.layer1 import Layer1Defense
        from defence.layer2 import Layer2Defense
        from defence.layer3 import Layer3Defense
        from defence.layer4 import Layer4Defense
        return DefensePipeline([
            Layer1Defense(),
            Layer2Defense(),
            Layer3Defense(),
            Layer4Defense(),
        ])

    if spec_str in (
        "1+2+3+4+5",
        "layer1+2+3+4+5",
        "layer1_2_3_4_5",
        "5layer",
        "5-layer",
        "5layers",
        "five_layer",
        "c4",
        "all",
        "full",
        "pipeline",
        "pipeline12345",
    ):
        from defence.layer1 import Layer1Defense
        from defence.layer2 import Layer2Defense
        from defence.layer3 import Layer3Defense
        from defence.layer4 import Layer4Defense
        from defence.layer5 import Layer5Defense
        return DefensePipeline([
            Layer1Defense(),
            Layer2Defense(),
            Layer3Defense(),
            Layer4Defense(),
            Layer5Defense(),
        ])

    if spec_str in ("without_l1_l2", "without_l1+l2", "no_l1_l2", "3+4", "l3+l4", "layer3+4", "layer3_4"):
        from defence.layer3 import Layer3Defense
        from defence.layer4 import Layer4Defense
        return DefensePipeline([Layer3Defense(), Layer4Defense()])

    if spec_str in ("without_l1_l2_with_l5", "3+4+5", "l3+l4+l5", "layer3+4+5"):
        from defence.layer3 import Layer3Defense
        from defence.layer4 import Layer4Defense
        from defence.layer5 import Layer5Defense
        return DefensePipeline([Layer3Defense(), Layer4Defense(), Layer5Defense()])

    # General arbitrary combination fallback: e.g. "1+4", "2+3", "3+5"
    if "+" in spec_str:
        tokens = [t.strip().replace("layer", "").replace("l", "") for t in spec_str.split("+") if t.strip()]
        layer_map = {
            "1": ("defence.layer1", "Layer1Defense"),
            "2": ("defence.layer2", "Layer2Defense"),
            "3": ("defence.layer3", "Layer3Defense"),
            "4": ("defence.layer4", "Layer4Defense"),
            "5": ("defence.layer5", "Layer5Defense"),
        }
        if all(t in layer_map for t in tokens):
            built_layers = []
            for t in tokens:
                mod_name, cls_name = layer_map[t]
                import importlib
                mod = importlib.import_module(mod_name)
                cls = getattr(mod, cls_name)
                built_layers.append(cls())
            return DefensePipeline(built_layers)

    raise ValueError(
        f"Unknown defense specification: '{spec}'. Expected 0, 1, 2, 3, 4, 5, 'layer1'..'layer5', 'without_l1_l2', '4layer', '5layer', or BaseDefense instance."
    )

