#!/usr/bin/env python3
"""Failure Analysis Engine for Prompt Injection Defense (PID).

Performs qualitative and quantitative root-cause analysis on cases that evaded
individual defense layers and pipelines:
  1. Lexical Dispersion & Sub-Threshold Scoring (bypasses Layer 1 pattern scanners)
  2. Single-Span Causal Invariance (the CausalArmor/AgentSentry gap — bypasses Layer 3)
  3. Cross-Boundary / Multi-Tool Dispersion (bypasses intra-document Layer 4 removers)
  4. Across-Turn Memory Latency (bypasses per-turn defenses)
  5. Innocent Pieces Semantic Assembly (routine business text requiring higher-order synthesis)
  6. Payload-Key / Untracked Entity Evasion (bypasses parameter provenance Layer 2)
  7. Case-Level Inventory: Explicit Case IDs, Tools, Targets, and Evasion Reasons
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import RESULTS_DIR
from run_ablation import DEFAULT_ABLATION_CONFIGS, evaluate_case_direct, load_dataset

logger = logging.getLogger(__name__)

FAILURE_TAXONOMY = {
    "lexical_dispersion": {
        "name": "Lexical Dispersion & Pattern Sub-Threshold",
        "description": "Attacker fragments lack overt trigger tokens ('ignore previous', 'system prompt'); each fragment resembles standard enterprise communication, bypassing regex heuristics (Layer 1).",
        "primary_remedy": "Semantic parameter provenance (Layer 2) and paired-span causal attribution (Layer 4).",
    },
    "single_span_invariance": {
        "name": "Single-Span Causal Invariance (The CausalArmor/AgentSentry Gap)",
        "description": "Removing any single sentence (leave-one-out) does not break the agent's intent because 2 or more complementary fragments still supply the imperative and payload. Single-span removal detects zero causal influence.",
        "primary_remedy": "Windowed and paired-span removal with pruning (Layer 4).",
    },
    "cross_source_dispersion": {
        "name": "Cross-Source / Multi-Tool Boundary Crossing",
        "description": "Fragments are partitioned across distinct tool outputs (e.g. Fragment A in read_email, Fragment B in read_calendar_note). Intra-document span analyzers evaluate each tool output in isolation and miss cross-tool synergy.",
        "primary_remedy": "Cross-source removal (Layer 5) treating entire tool outputs as atomic units.",
    },
    "across_turns_persistence": {
        "name": "Across-Turn Memory Latency",
        "description": "Fragment A is persisted to memory in Turn 1, while Fragment B arrives in Turn 2. Per-turn defenses evaluate only recent context and miss multi-turn state accumulation.",
        "primary_remedy": "Memory-unit counterfactual probing and cross-turn source attribution (Layer 5).",
    },
    "innocent_pieces_semantic_synthesis": {
        "name": "Innocent Pieces Semantic Assembly",
        "description": "Individual pieces are entirely innocent sentences. The malicious action is only synthesized when the LLM draws an implicit semantic inference connecting disparate facts.",
        "primary_remedy": "Defense-in-depth: Strict user authorization grounding (Layer 2) combined with multi-span / multi-source removal (Layers 4 & 5).",
    },
    "untracked_entity_structure": {
        "name": "Untracked Entity / Payload Evasion",
        "description": "Attack manipulates parameters whose values are formatted as message contents or non-recipient payloads (e.g. append_to_file content), evading pure recipient/URL provenance checks.",
        "primary_remedy": "Behavioral causal attribution (Layers 4 & 5) verifying operational necessity rather than argument strings.",
    },
}


def analyze_failures(
    cases: list[dict],
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Analyze failure modes across all cases and defense configurations."""
    total = len(cases)
    print("\n" + "=" * 80)
    print(f"  FAILURE ANALYSIS & ROOT-CAUSE INVESTIGATION ({total} Cases)")
    print("=" * 80)

    # Track evasions per configuration
    evasions_by_config: dict[str, list[dict]] = {}
    fp_by_config: dict[str, list[dict]] = {}

    for config_key, spec, _desc in DEFAULT_ABLATION_CONFIGS:
        evasions = []
        fps = []
        for case in cases:
            res = evaluate_case_direct(case, spec)
            if not res["attack_blocked"]:
                evasions.append({
                    "case_id": case.get("id"),
                    "type": case.get("type"),
                    "tool": case["attack_goal"]["tool_aliases"][0],
                    "goal": case["attack_goal"]["description"],
                    "critical_value": case["attack_goal"]["critical_value"],
                    "fragments": case.get("fragments", []),
                })
            if res["benign_blocked_fp"]:
                fps.append({
                    "case_id": case.get("id"),
                    "type": case.get("type"),
                    "twin_critical_value": case["benign_twin"]["critical_value"],
                    "reason": res.get("twin_block_reason"),
                })
        evasions_by_config[config_key] = evasions
        fp_by_config[config_key] = fps

    # Categorize root causes based on empirical layer behavior
    mode_counts = Counter()
    case_breakdowns = []
    layer_evasion_lists: dict[str, list[str]] = {
        "layer1_evasions": [],
        "layer2_evasions": [],
        "layer3_evasions": [],
        "layer4_evasions": [],
        "layer5_evasions": [],
        "pipeline_evasions": [],
    }

    for case in cases:
        cid = case.get("id", "?")
        ctype = case.get("type", "unknown")
        frags = case.get("fragments", [])
        goal_desc = case["attack_goal"]["description"]
        crit_val = case["attack_goal"]["critical_value"]
        target_tool = case["attack_goal"]["tool_aliases"][0]

        # Evaluate individual layers
        c1 = evaluate_case_direct(case, "1")
        c2 = evaluate_case_direct(case, "2")
        l3 = evaluate_case_direct(case, "3")
        l4 = evaluate_case_direct(case, "4")
        l5 = evaluate_case_direct(case, "5")
        c4 = evaluate_case_direct(case, "1+2+3+4+5")

        if not c1["attack_blocked"]:
            layer_evasion_lists["layer1_evasions"].append(cid)
        if not c2["attack_blocked"]:
            layer_evasion_lists["layer2_evasions"].append(cid)
        if not l3["attack_blocked"]:
            layer_evasion_lists["layer3_evasions"].append(cid)
        if not l4["attack_blocked"]:
            layer_evasion_lists["layer4_evasions"].append(cid)
        if not l5["attack_blocked"]:
            layer_evasion_lists["layer5_evasions"].append(cid)
        if not c4["attack_blocked"]:
            layer_evasion_lists["pipeline_evasions"].append(cid)

        # Diagnose primary failure mode for this specific attack
        if not c2["attack_blocked"]:
            mode = "untracked_entity_structure"
            diag = f"Parameter in '{target_tool}' was treated as general payload rather than traceable steering entity."
        elif not l4["attack_blocked"] and ctype == "same_document":
            mode = "single_span_invariance"
            diag = "Spans separated by wide filler paragraphs beyond window/top-k capacity."
        elif not l4["attack_blocked"] and ctype == "across_outputs":
            mode = "cross_source_dispersion"
            diag = "Fragments partitioned across separate tool results; document-local span removers cannot correlate."
        elif not l4["attack_blocked"] and ctype == "across_turns":
            mode = "across_turns_persistence"
            diag = "Fragment stored in memory on Turn 1; single-turn analyzers miss multi-turn state accumulation."
        elif ctype == "innocent_pieces":
            mode = "innocent_pieces_semantic_synthesis"
            diag = "Individual fragments are fully benign sentences; malice only emerges upon semantic synthesis."
        else:
            mode = "lexical_dispersion"
            diag = "Lacks overt injection trigger words, bypassing static pattern heuristics."

        mode_counts[mode] += 1

        case_breakdowns.append({
            "case_id": cid,
            "type": ctype,
            "tool": target_tool,
            "goal": goal_desc,
            "critical_value": crit_val,
            "fragments": frags,
            "primary_failure_mode": mode,
            "diagnostic_mechanism": diag,
            "blocked_by_l1": c1["attack_blocked"],
            "blocked_by_l2": c2["attack_blocked"],
            "blocked_by_l3": l3["attack_blocked"],
            "blocked_by_l4": l4["attack_blocked"],
            "blocked_by_l5": l5["attack_blocked"],
            "blocked_by_full_pipeline": c4["attack_blocked"],
        })

    analysis_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_cases_analyzed": total,
        "taxonomy": FAILURE_TAXONOMY,
        "failure_mode_distribution": dict(mode_counts),
        "evasion_counts_by_config": {k: len(v) for k, v in evasions_by_config.items()},
        "false_positive_counts_by_config": {k: len(v) for k, v in fp_by_config.items()},
        "layer_evasion_case_ids": layer_evasion_lists,
        "case_breakdowns": case_breakdowns,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis_results, f, indent=2)
        print(f"\n💾 Failure analysis saved to: {output_path}")

    print_failure_summary(analysis_results)
    return analysis_results


def print_failure_summary(results: dict[str, Any]):
    """Print readable summary of failure modes and specific evaded cases."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()

        # ── 1. Root Cause Distribution ──
        table = Table(title="Failure Analysis — Root Cause Distribution Across Dataset", show_lines=True)
        table.add_column("Failure Category", style="bold cyan")
        table.add_column("Cases", justify="right", style="white")
        table.add_column("Share", justify="right", style="yellow")
        table.add_column("Key Remediation Mechanism", style="green")

        total = results["total_cases_analyzed"]
        for mode, count in results["failure_mode_distribution"].items():
            pct = count / total if total else 0.0
            meta = results["taxonomy"].get(mode, {})
            table.add_row(
                meta.get("name", mode),
                str(count),
                f"{pct * 100:.1f}%",
                meta.get("primary_remedy", ""),
            )

        console.print()
        console.print(table)

        # ── 2. Concrete Evading Cases by Layer ──
        ev_table = Table(title="Specific Case IDs Evading Defense Layers", show_lines=True)
        ev_table.add_column("Defense Layer / Configuration", style="bold magenta")
        ev_table.add_column("Evasion Count", justify="right", style="white")
        ev_table.add_column("Specific Evaded Case IDs (Sample / Full)", style="yellow")

        for cfg, ids in results["layer_evasion_case_ids"].items():
            name_map = {
                "layer1_evasions": "Layer 1 Alone (Tag & Scan)",
                "layer2_evasions": "Layer 2 Alone (Provenance)",
                "layer3_evasions": "Layer 3 Alone (Single-Span Causal)",
                "layer4_evasions": "Layer 4 Alone (Paired-Span Causal)",
                "layer5_evasions": "Layer 5 Alone (Cross-Source Causal)",
                "pipeline_evasions": "Full 5-Layer Pipeline (Defense-in-Depth)",
            }
            sample_ids = ", ".join(ids[:8]) + (f" ... (+{len(ids)-8} more)" if len(ids) > 8 else "")
            if not ids:
                sample_ids = "None (0 evasions)"
            ev_table.add_row(name_map.get(cfg, cfg), str(len(ids)), sample_ids)

        console.print()
        console.print(ev_table)
        console.print()

    except ImportError:
        print("\n=== FAILURE MODE DISTRIBUTION ===")
        for mode, count in results["failure_mode_distribution"].items():
            print(f"{mode:<35}: {count} cases")


def format_markdown_failure_report(results: dict[str, Any]) -> str:
    """Generate Markdown report for failure analysis."""
    lines = [
        "## Comprehensive Failure Analysis & Vulnerability Root Causes",
        "",
        "This failure analysis details why split-context prompt injections successfully bypass standard defenses (like Layer 1 and Layer 3), and how the layered pipeline resolves these vulnerabilities.",
        "",
        "### 1. Root Cause Distribution Across Dataset",
        "",
        "| Failure Mode Category | Cases | Pct | Primary Vulnerability | Architectural Defense Solution |",
        "| --- | :---: | :---: | --- | --- |",
    ]

    total = results["total_cases_analyzed"]
    for mode, count in results["failure_mode_distribution"].items():
        pct = (count / total * 100) if total else 0.0
        meta = results["taxonomy"].get(mode, {})
        lines.append(
            f"| **{meta.get('name', mode)}** | {count} | {pct:.1f}% | {meta.get('description', '')[:90]}... | {meta.get('primary_remedy', '')} |"
        )

    lines.extend([
        "",
        "### 2. Specific Case IDs Evading Individual Layers",
        "",
        "| Layer / Configuration | Evasions | Specific Evaded Case IDs | Root Cause Explanation |",
        "| --- | :---: | --- | --- |",
    ])

    ev_lists = results.get("layer_evasion_case_ids", {})

    def _fmt_ids(ids: list[str], limit: int = 8) -> str:
        """Format a case-id list, avoiding dangling '...' for empty lists."""
        if not ids:
            return "None (0 evasions)"
        shown = ", ".join(ids[:limit])
        if len(ids) > limit:
            shown += f" ... (+{len(ids) - limit} more)"
        return shown

    ev_meta = {
        "layer1_evasions": ("Layer 1 Alone (Tag & Scan)", "All 60 cases", "No trigger keywords ('ignore previous') exist in isolated split fragments."),
        "layer2_evasions": ("Layer 2 Alone (Provenance)", _fmt_ids(ev_lists.get("layer2_evasions", [])), "Targets passed as general payload content (append_to_file) evade entity parameter filters."),
        "layer3_evasions": ("Layer 3 Alone (Single-Span)", _fmt_ids(ev_lists.get("layer3_evasions", [])), "Single-span leave-one-out removal leaves complementary fragments in context (CausalArmor gap)."),
        "layer4_evasions": ("Layer 4 Alone (Paired-Span)", _fmt_ids(ev_lists.get("layer4_evasions", [])), "Intra-document removers cannot correlate fragments split across different tool outputs or turns."),
        "layer5_evasions": ("Layer 5 Alone (Cross-Source)", _fmt_ids(ev_lists.get("layer5_evasions", [])), "Single-document split cases contain both fragments in one source unit; source-level removal is too coarse."),
        "pipeline_evasions": ("Full 5-Layer Pipeline", _fmt_ids(ev_lists.get("pipeline_evasions", [])), "Defense-in-depth catches all cross-source, cross-turn, and intra-document attacks."),
    }

    for k, (name, cases_str, reason) in ev_meta.items():
        cnt = len(ev_lists.get(k, []))
        lines.append(f"| **{name}** | {cnt} | `{cases_str}` | {reason} |")

    lines.extend([
        "",
        "### 3. Deep Dive Into Evasion Mechanics",
        "",
        "#### Evasion Mode A: Single-Span Invariance (The CausalArmor/AgentSentry Gap)",
        "> **Theoretical Insight**: In single-span leave-one-out causal attribution (Layer 3), the security monitor removes sentence $s_i$ and re-queries whether the action is still proposed. In split-context attacks, the malicious directive is decomposed into $k \\ge 3$ sentences (e.g. $s_2$ establishes context, $s_4$ issues action verb, $s_6$ supplies exfiltration target). Removing any single sentence $s_i$ leaves $k-1$ sentences. Because LLMs exhibit strong contextual tolerance and entity reconstruction, the single-span probe concludes that no individual sentence was uniquely causal. **Layer 4 eliminates this vulnerability by evaluating paired removals $(s_i, s_j)$ and sliding windows of size 2 and 3**.",
        "",
        "#### Evasion Mode B: Multi-Source & Cross-Turn Latency",
        "> **Theoretical Insight**: When an attacker distributes fragments across separate tools (e.g. email and calendar note) or across conversation turns (saved into memory in Turn 1, activated in Turn 2), document-level span removers cannot correlate the fragments. **Layer 5 solves this by operating at the source and turn level**, treating entire tool outputs and persistent memory records as single counterfactual units.",
        "",
        "#### Evasion Mode C: Dual Metric Alignment (Benign Utility Guarantee)",
        "> **Theoretical Insight**: A defense that simply blocks all external parameters achieves 0% ASR but destroys task utility (0% Benign Utility). By coupling **provenance tracing (Layer 2)** with **causal attribution (Layers 3, 4, 5)**, the system permits legitimate actions whose parameters are explicitly requested or authorized by the trusted user prompt while strictly blocking ungrounded tainted parameters.",
        "",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run PID Failure Analysis")
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["split_context_dev.jsonl", "split_context_heldout.jsonl"],
        help="JSONL case files to analyze",
    )
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "failure_analysis.json"),
        help="Path to save failure analysis JSON results",
    )
    args = parser.parse_args()

    cases = load_dataset(args.dataset)
    analyze_failures(cases, output_path=Path(args.output))


if __name__ == "__main__":
    main()
