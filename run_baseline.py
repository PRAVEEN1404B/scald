#!/usr/bin/env python3
"""Master baseline execution script — Phase 0.

Runs all benchmarks and produces the unified results/baseline_asr.json
that will be reused for before/after comparison in every later defense phase.

Usage:
    python run_baseline.py                    # Full InjecAgent run
    python run_baseline.py --smoke            # Quick smoke test (5 cases)
    python run_baseline.py --injecagent-only  # Only InjecAgent (default)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import (
    RESULTS_DIR,
    add_reproducibility_args,
    apply_reproducibility_args,
    get_llm_settings,
    get_run_metadata,
)

logger = logging.getLogger(__name__)


def print_summary_table(results: dict):
    """Print a rich summary table of all benchmark results."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console(color_system=None)
        table = Table(title="Phase 0 — Undefended Baseline ASR", show_lines=True)
        table.add_column("Benchmark", style="bold cyan")
        table.add_column("Category", style="white")
        table.add_column("Cases", justify="right")
        table.add_column("Injected", justify="right", style="red")
        table.add_column("ASR", justify="right", style="bold red")

        # InjecAgent results
        if "injecagent" in results:
            ij = results["injecagent"]
            if "overall_asr" in ij:
                table.add_row(
                    "InjecAgent",
                    "overall",
                    str(ij.get("overall_total", "?")),
                    str(ij.get("overall_successes", "?")),
                    f"{ij['overall_asr']:.1%}",
                )
                for var_name, var_data in ij.get("variants", {}).items():
                    if isinstance(var_data, dict) and "asr" in var_data:
                        table.add_row(
                            f"  └ {var_name}",
                            "overall",
                            str(var_data.get("total_cases", "?")),
                            str(var_data.get("successful_injections", "?")),
                            f"{var_data['asr']:.1%}",
                        )
                        for cat_name, cat_data in var_data.get("categories", {}).items():
                            table.add_row(
                                "",
                                cat_name,
                                str(cat_data.get("total", "?")),
                                str(cat_data.get("successful", "?")),
                                f"{cat_data['asr']:.1%}",
                            )

        console.print()
        console.print(table)
        console.print()

    except ImportError:
        # Fallback if rich is not installed
        print("\n" + "=" * 70)
        print("  BASELINE ASR RESULTS")
        print("=" * 70)
        print(json.dumps(results, indent=2))


def run_demo():
    """Run the vulnerability demo as a sanity check."""
    print("\n" + "=" * 70)
    print("  Step 1: Vulnerability Demo (Proof of Concept)")
    print("=" * 70)

    try:
        # Import and run the demo
        sys.path.insert(0, str(Path(__file__).parent / "demo"))
        from demo.run_demo import main as demo_main
        demo_main()
        return True
    except Exception as e:
        logger.error("Demo failed: %s", e)
        print(f"Demo failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 0 baseline benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test (5 cases per benchmark)")
    parser.add_argument("--injecagent-only", action="store_true",
                        help="Run only InjecAgent benchmark (default)")
    parser.add_argument("--skip-demo", action="store_true",
                        help="Skip the vulnerability demo")
    default_model, default_base_url, default_api_key = get_llm_settings()
    parser.add_argument("--model", default=None,
                        help=f"Model to use (default: {default_model})")
    parser.add_argument("--base-url", default=None,
                        help=f"API base URL (default: {default_base_url})")
    parser.add_argument("--api-key", default=None,
                        help="API key for LLM endpoint")
    parser.add_argument("--variants", nargs="+",
                        default=["base", "enhanced"],
                        help="InjecAgent variants to run (default: base + enhanced)")
    parser.add_argument("--max-cases", type=int, default=0,
                        help="Max InjecAgent cases per variant (default: 0 = all cases, i.e. full baseline). "
                             "Use a positive number for a quick partial run.")
    add_reproducibility_args(parser)
    args = parser.parse_args()
    apply_reproducibility_args(args)

    model = args.model or default_model
    base_url = args.base_url or default_base_url
    api_key = args.api_key or default_api_key

    # Normalize variants (supports both comma-separated and space-separated)
    variants = []
    for v in args.variants:
        for part in v.split(","):
            part = part.strip()
            if part and part not in variants:
                variants.append(part)

    run_injecagent = True

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 70)
    print("  Prompt Injection Defense — Phase 0 Baseline")
    print(f"  Model: {model}")
    print(f"  Time:  {datetime.now().isoformat()}")
    print("=" * 70)

    output_path = RESULTS_DIR / "baseline_asr.json"
    # Start fresh: never merge with stale results from a previous run/model.
    # (This previously left an outdated AgentDojo section in the file.)
    combined_results: dict[str, Any] = {}

    max_cases = 5 if args.smoke else args.max_cases
    is_full_dataset = not args.smoke and max_cases == 0

    combined_results.update({
        "phase": 0,
        "model": model,
        "defense": "none",
        "run_metadata": get_run_metadata(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "smoke_test": args.smoke,
        "is_full_dataset": is_full_dataset,
        "max_cases_per_variant": None if max_cases == 0 else max_cases,
        "variants_requested": variants,
    })

    if not is_full_dataset:
        print("\nWARNING: PARTIAL RUN: this is NOT a full baseline.")
        print("   Use --max-cases 0 (default) for the complete InjecAgent set.")

    # Step 1: Demo
    if not args.skip_demo and not args.injecagent_only:
        run_demo()

    # Step 2: InjecAgent
    if run_injecagent:
        print("\n" + "=" * 70)
        print("  Step 2: InjecAgent Benchmark")
        print("=" * 70)

        try:
            from benchmarks.injecagent_runner import run_injecagent_benchmark

            ij_results = run_injecagent_benchmark(
                variants=variants,
                model=model,
                base_url=base_url,
                api_key=api_key,
                max_cases=max_cases,
            )
            combined_results["injecagent"] = ij_results

        except FileNotFoundError as e:
            logger.error("InjecAgent data not found: %s", e)
            print(f"InjecAgent data not found: {e}")
            print("   Clone with: git clone https://github.com/uiuc-kang-lab/InjecAgent")
            combined_results["injecagent"] = {"error": str(e)}
        except Exception as e:
            logger.error("InjecAgent benchmark failed: %s", e)
            print(f"InjecAgent benchmark failed: {e}")
            combined_results["injecagent"] = {"error": str(e)}

    # Step 3: Save combined results
    output_path = RESULTS_DIR / "baseline_asr.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2)

    print(f"\nCombined results saved to: {output_path}")

    # Step 4: Print summary
    print_summary_table(combined_results)

    print("\nPhase 0 baseline complete.")
    print(f"   Results file: {output_path}")
    print("   This file will be reused for before/after comparison in later phases.")


if __name__ == "__main__":
    main()
