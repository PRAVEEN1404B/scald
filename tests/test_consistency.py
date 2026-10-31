"""Regression tests for cross-stage consistency.

Ablation, coverage matrix, and failure analysis used to re-evaluate the same
(case, configuration) pairs independently. Because Layers 3-5 call an LLM, the
report could show e.g. 20% ASR in one table and 56% in another for the same
configuration. These tests pin the single-evaluation guarantee.
"""
import pytest

import coverage_matrix
import failure_analysis
import run_ablation

CASES = [
    {
        "id": "T1-001",
        "type": "same_document",
        "fragments": ["a", "b", "c"],
        "attack_goal": {
            "tool_aliases": ["send_email"],
            "description": "email outside",
            "critical_value": "attacker@example.test",
        },
        "benign_twin": {"critical_value": "dana.roy@example.test"},
    },
    {
        "id": "T3-001",
        "type": "across_turns",
        "fragments": ["d", "e", "f"],
        "attack_goal": {
            "tool_aliases": ["share_file"],
            "description": "share outside",
            "critical_value": "attacker@example.test",
        },
        "benign_twin": {"critical_value": "dana.roy@example.test"},
    },
]


@pytest.fixture
def stub_evaluator(monkeypatch):
    """Deterministic replacement for the LLM-backed evaluator, with a call counter."""
    calls = []

    def fake_evaluate(case, spec):
        spec = str(spec)
        calls.append((case["id"], spec))

        if spec == "always_block":
            blocked = True
        elif spec in {"none", "always_allow"}:
            blocked = False
        else:
            blocked = (sum(case["id"].encode()) + len(spec)) % 3 == 0

        benign_blocked = spec == "always_block"
        return {
            "case_id": case["id"],
            "type": case["type"],
            "attack_succeeded": (not blocked) and case["id"].endswith("1"),
            "attack_blocked": blocked,
            "block_reason": "blocked" if blocked else None,
            "benign_task_completed": not benign_blocked,
            "benign_blocked_fp": benign_blocked,
            "twin_block_reason": "blocked" if benign_blocked else None,
            "attack_probes_used": 0 if blocked else 1,
            "twin_probes_used": 0,
            "probes_used": 0 if blocked else 1,
        }

    monkeypatch.setattr(run_ablation, "evaluate_case_direct", fake_evaluate)
    run_ablation.reset_eval_cache()
    yield calls
    run_ablation.reset_eval_cache()


def test_stages_reuse_a_single_evaluation_pass(stub_evaluator):
    ablation = run_ablation.run_ablation_study(CASES)
    calls_after_ablation = len(stub_evaluator)
    assert calls_after_ablation == len(run_ablation.DEFAULT_ABLATION_CONFIGS) * len(CASES)

    coverage = coverage_matrix.generate_coverage_matrix(CASES, ablation_results=ablation)
    # Every coverage configuration was already evaluated by the ablation stage.
    assert len(stub_evaluator) == calls_after_ablation

    failure = failure_analysis.analyze_failures(CASES, ablation_results=ablation)
    # Only the standalone L1/L2/L5 rows are new (L3/L4/full pipeline are reused).
    assert len(stub_evaluator) == calls_after_ablation + 3 * len(CASES)

    # 1. Ablation vs coverage matrix overall counts must agree exactly.
    for cfg, data in ablation["configurations"].items():
        assert (
            coverage["rows"][cfg]["cells"]["overall"]["blocked"]
            == data["attacks_blocked"]
        ), cfg
        assert (
            coverage["rows"][cfg]["cells"]["overall"]["attack_succeeded"]
            == data["attack_successes"]
        ), cfg

    # 2. Failure analysis evasion counts must be the complement of the same blocks.
    for cfg, data in ablation["configurations"].items():
        assert failure["evasion_counts_by_config"][cfg] == len(CASES) - data["attacks_blocked"], cfg

    # 3. Standalone layer rows in the failure table must equal the matching
    #    ablation configurations, not a second evaluation.
    for layer_key, ablation_key in [
        ("layer3_evasions", "l3_alone"),
        ("layer4_evasions", "l4_alone"),
        ("pipeline_evasions", "full_pipeline"),
    ]:
        expected = len(CASES) - ablation["configurations"][ablation_key]["attacks_blocked"]
        assert len(failure["layer_evasion_case_ids"][layer_key]) == expected, layer_key


def test_cache_prevents_reevaluation_without_precomputed(stub_evaluator):
    ablation = run_ablation.run_ablation_study(CASES)
    before = len(stub_evaluator)

    # No precomputed index: the in-process cache still supplies every result.
    coverage_matrix.generate_coverage_matrix(CASES)
    assert len(stub_evaluator) == before

    failure_analysis.analyze_failures(CASES)
    assert len(stub_evaluator) == before + 3 * len(CASES)

    assert ablation["total_cases"] == len(CASES)


def test_precomputed_index_is_keyed_by_spec_and_case(stub_evaluator):
    ablation = run_ablation.run_ablation_study(CASES)
    index = run_ablation.index_per_case_results(ablation)

    assert ("T1-001", "none") in index
    assert ("T1-001", "1+2+3+4+5") in index
    assert run_ablation.precomputed_matches(index, CASES) is True
    assert run_ablation.precomputed_matches(index, CASES + [{"id": "T9-999"}]) is False


def test_mismatched_ablation_results_trigger_reevaluation(stub_evaluator, caplog):
    ablation = run_ablation.run_ablation_study(CASES)
    before = len(stub_evaluator)

    other_cases = [dict(CASES[0], id="T4-999")]
    coverage_matrix.generate_coverage_matrix(other_cases, ablation_results=ablation)

    # The stale index is rejected, so the new case is evaluated for every config.
    assert len(stub_evaluator) == before + len(run_ablation.DEFAULT_ABLATION_CONFIGS)
