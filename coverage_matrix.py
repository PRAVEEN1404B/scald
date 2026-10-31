#!/usr/bin/env python3
"""Coverage Matrix Generator (Layers × Attack Types) for PID.

Cross-tabulates defense configurations against the 4 split-context attack types:
  1. same_document   – Goal split across 3+ sentences in one document (Target: Layer 4)
  2. across_outputs  – Fragment A in email, Fragment B in calendar note (Target: Layer 5)
  3. across_turns    – One fragment saved to memory, next arrives later (Target: Layer 5)
  4. innocent_pieces – Normal sentences that combine into an attack (Target: Layer 4/5)
  5. Overall         – Combined dataset summary

Metrics reported per cell:
  - Block / Detection Rate (%) with Wilson 95% Confidence Intervals
  - Attack Success Rate (ASR) with Wilson 95% Confidence Intervals
  - Benign Utility (BU) / Task Completion Rate with Wilson 95% Confidence Intervals
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import RESULTS_DIR
from metrics.wilson import wilson_interval
from run_ablation import (
    DEFAULT_ABLATION_CONFIGS,
    index_per_case_results,
    load_dataset,
    precomputed_matches,
    resolve_case_result,
)

logger = logging.getLogger(__name__)

ATTACK_TYPES = [
    "same_document",
    "across_outputs",
    "across_turns",
    "innocent_pieces",
]


def generate_coverage_matrix(
    cases: list[dict],
    configs: list[tuple[str, str, str]] | None = None,
    output_path: Path | None = None,
    ablation_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate the full layers × attack types coverage matrix.

    When ``ablation_results`` is supplied, per-case outcomes are reused from that
    single evaluation pass so the coverage table can never contradict the
    ablation table. Missing configurations are evaluated once and cached.
    """
    eval_configs = configs or DEFAULT_ABLATION_CONFIGS
    precomputed = index_per_case_results(ablation_results) if ablation_results else None
    if precomputed and not precomputed_matches(precomputed, cases):
        logger.warning("Ablation results do not cover the current dataset; re-evaluating.")
        precomputed = None

    matrix_data: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(cases),
        "attack_types": ATTACK_TYPES + ["overall"],
        "rows": {},
    }

    # Group cases by type
    cases_by_type: dict[str, list[dict]] = {t: [] for t in ATTACK_TYPES}
    for c in cases:
        t = c.get("type", "unknown")
        if t in cases_by_type:
            cases_by_type[t].append(c)
        else:
            cases_by_type.setdefault(t, []).append(c)

    print("\n" + "=" * 80)
    print(f"  COVERAGE MATRIX: DEFENSE LAYERS × ATTACK TYPES ({len(cases)} Cases)")
    print("=" * 80)

    for config_key, spec, desc in eval_configs:
        matrix_data["rows"][config_key] = {
            "name": config_key,
            "spec": spec,
            "description": desc,
            "cells": {},
        }

        # Evaluate each attack type subset
        for atype in ATTACK_TYPES:
            type_cases = cases_by_type.get(atype, [])
            total = len(type_cases)
            if total == 0:
                continue

            blocked = 0
            attack_succeeded = 0
            utility_success = 0
            fp_count = 0

            for c in type_cases:
                res = resolve_case_result(c, spec, precomputed)
                if res["attack_blocked"]:
                    blocked += 1
                if res["attack_succeeded"]:
                    attack_succeeded += 1
                if res["benign_task_completed"]:
                    utility_success += 1
                if res["benign_blocked_fp"]:
                    fp_count += 1

            matrix_data["rows"][config_key]["cells"][atype] = {
                "total": total,
                "blocked": blocked,
                "attack_succeeded": attack_succeeded,
                "utility_success": utility_success,
                "fp_count": fp_count,
                "block_rate": wilson_interval(blocked, total).to_dict(),
                "asr": wilson_interval(attack_succeeded, total).to_dict(),
                "utility": wilson_interval(utility_success, total).to_dict(),
                "fpr": wilson_interval(fp_count, total).to_dict(),
            }

        # Overall summary column
        all_total = len(cases)
        all_blocked = sum(
            matrix_data["rows"][config_key]["cells"][t]["blocked"]
            for t in ATTACK_TYPES
            if t in matrix_data["rows"][config_key]["cells"]
        )
        all_succeeded = sum(
            matrix_data["rows"][config_key]["cells"][t]["attack_succeeded"]
            for t in ATTACK_TYPES
            if t in matrix_data["rows"][config_key]["cells"]
        )
        all_utility = sum(
            matrix_data["rows"][config_key]["cells"][t]["utility_success"]
            for t in ATTACK_TYPES
            if t in matrix_data["rows"][config_key]["cells"]
        )
        all_fp = sum(
            matrix_data["rows"][config_key]["cells"][t]["fp_count"]
            for t in ATTACK_TYPES
            if t in matrix_data["rows"][config_key]["cells"]
        )

        matrix_data["rows"][config_key]["cells"]["overall"] = {
            "total": all_total,
            "blocked": all_blocked,
            "attack_succeeded": all_succeeded,
            "utility_success": all_utility,
            "fp_count": all_fp,
            "block_rate": wilson_interval(all_blocked, all_total).to_dict(),
            "asr": wilson_interval(all_succeeded, all_total).to_dict(),
            "utility": wilson_interval(all_utility, all_total).to_dict(),
            "fpr": wilson_interval(all_fp, all_total).to_dict(),
        }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(matrix_data, f, indent=2)
        print(f"\n💾 Coverage matrix saved to: {output_path}")

    print_matrix_tables(matrix_data)
    return matrix_data


def print_matrix_tables(data: dict[str, Any]):
    """Print Rich and Markdown formatted coverage matrices."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()

        # ── 1. Block / Detection Rate Matrix ──
        table_block = Table(
            title="Coverage Matrix — Detection / Block Rate by Attack Type (Wilson 95% CI)",
            show_lines=True,
        )
        table_block.add_column("Configuration", style="bold cyan")
        table_block.add_column("Layers", style="white")
        for atype in ATTACK_TYPES:
            table_block.add_column(atype.replace("_", " ").title(), justify="center")
        table_block.add_column("OVERALL", justify="center", style="bold green")

        for key, row in data["rows"].items():
            cells = row["cells"]
            row_vals = [key, row["spec"]]
            for atype in ATTACK_TYPES:
                if atype in cells:
                    r = cells[atype]["block_rate"]["rate"]
                    row_vals.append(f"{r * 100:.1f}%")
                else:
                    row_vals.append("—")
            ov = cells.get("overall", {}).get("block_rate", {})
            row_vals.append(f"{ov.get('rate', 0.0) * 100:.1f}%")
            table_block.add_row(*row_vals)

        console.print()
        console.print(table_block)

        # ── 2. Benign Utility Matrix ──
        table_util = Table(
            title="Coverage Matrix — Benign Utility / Task Completion Rate (Wilson 95% CI)",
            show_lines=True,
        )
        table_util.add_column("Configuration", style="bold cyan")
        table_util.add_column("Layers", style="white")
        for atype in ATTACK_TYPES:
            table_util.add_column(atype.replace("_", " ").title(), justify="center")
        table_util.add_column("OVERALL", justify="center", style="bold blue")

        for key, row in data["rows"].items():
            cells = row["cells"]
            row_vals = [key, row["spec"]]
            for atype in ATTACK_TYPES:
                if atype in cells:
                    u = cells[atype]["utility"]["rate"]
                    row_vals.append(f"{u * 100:.1f}%")
                else:
                    row_vals.append("—")
            ov_u = cells.get("overall", {}).get("utility", {})
            row_vals.append(f"{ov_u.get('rate', 0.0) * 100:.1f}%")
            table_util.add_row(*row_vals)

        console.print()
        console.print(table_util)
        console.print()

    except ImportError:
        print("\n=== COVERAGE MATRIX (Detection Rate %) ===")
        for key, row in data["rows"].items():
            cells = row["cells"]
            ov = cells.get("overall", {}).get("block_rate", {}).get("formatted", "")
            print(f"{key:<18} | Overall Block: {ov}")


def format_markdown_coverage_matrix(data: dict[str, Any]) -> str:
    """Render the coverage matrix as a clean GitHub-style Markdown table."""
    headers = ["Configuration", "Layers", "Same Document", "Across Outputs", "Across Turns", "Innocent Pieces", "OVERALL (Block Rate / ASR / Utility)"]
    md = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]

    for key, row in data["rows"].items():
        cells = row["cells"]
        line = [f"`{key}`", f"`{row['spec']}`"]
        for atype in ATTACK_TYPES:
            if atype in cells:
                br = cells[atype]["block_rate"]["rate"] * 100
                asr = cells[atype]["asr"]["rate"] * 100
                bu = cells[atype]["utility"]["rate"] * 100
                line.append(f"**{br:.0f}%** blk <br><sub>ASR {asr:.0f}% | BU {bu:.0f}%</sub>")
            else:
                line.append("—")
        ov_b = cells.get("overall", {}).get("block_rate", {}).get("rate", 0.0) * 100
        ov_a = cells.get("overall", {}).get("asr", {}).get("rate", 0.0) * 100
        ov_u = cells.get("overall", {}).get("utility", {}).get("rate", 0.0) * 100
        ov_ci = cells.get("overall", {}).get("asr", {})
        line.append(f"**{ov_b:.1f}%** Block <br><sub>ASR {ov_a:.1f}% [CI: {ov_ci.get('lower', 0)*100:.1f}%-{ov_ci.get('upper', 0)*100:.1f}%] <br> BU: {ov_u:.1f}%</sub>")
        md.append("| " + " | ".join(line) + " |")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Generate PID Coverage Matrix")
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["split_context_dev.jsonl", "split_context_heldout.jsonl"],
        help="JSONL case files to evaluate",
    )
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "coverage_matrix.json"),
        help="Path to save coverage matrix JSON results",
    )
    parser.add_argument(
        "--ablation-results",
        default=str(RESULTS_DIR / "ablation_results.json"),
        help="Reuse per-case outcomes from this ablation run (default: results/ablation_results.json). "
             "Pass an empty string to re-evaluate everything.",
    )
    args = parser.parse_args()

    cases = load_dataset(args.dataset)

    ablation_results = None
    if args.ablation_results:
        path = Path(args.ablation_results)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                ablation_results = json.load(f)
            print(f"♻️  Reusing per-case results from {path}")
        else:
            print(f"⚠️  {path} not found; evaluating every configuration from scratch.")

    generate_coverage_matrix(
        cases,
        output_path=Path(args.output),
        ablation_results=ablation_results,
    )


if __name__ == "__main__":
    main()
