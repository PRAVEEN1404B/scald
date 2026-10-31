"""Tests for configuration, reproducibility plumbing, and defence selection."""
import random

import pytest

import config
import defence
import defense  # compatibility shim

# ── Reasoning-model detection ────────────────────────────────────────────────

@pytest.mark.parametrize(
    "model",
    ["openai/gpt-oss-20b", "o1-mini", "o3", "deepseek-r1:7b", "my-reasoner"],
)
def test_is_reasoning_model_true(model):
    assert config.is_reasoning_model(model) is True


@pytest.mark.parametrize("model", ["qwen2.5:7b", "gpt-4o-mini", "llama3.1", "", None])
def test_is_reasoning_model_false(model):
    assert config.is_reasoning_model(model) is False


# ── Backend resolution ───────────────────────────────────────────────────────

def test_get_llm_settings_groq(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEYS", ["test-key"])
    monkeypatch.setattr(config, "GROQ_MODEL", "test-groq-model")
    monkeypatch.setattr(config, "GROQ_BASE_URL", "https://groq.example/v1")
    assert config.get_llm_settings("groq") == (
        "test-groq-model",
        "https://groq.example/v1",
        "test-key",
    )


def test_get_llm_settings_falls_back_to_ollama(monkeypatch):
    monkeypatch.setattr(config, "get_current_groq_key", lambda: "")
    monkeypatch.setattr(config, "OLLAMA_MODEL", "qwen-test")
    monkeypatch.setattr(config, "OLLAMA_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setattr(config, "OLLAMA_API_KEY", "ollama")
    assert config.get_llm_settings("groq") == (
        "qwen-test",
        "http://localhost:11434/v1",
        "ollama",
    )


def test_backend_name_is_normalized(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEYS", ["k"])
    monkeypatch.setattr(config, "GROQ_MODEL", "m")
    monkeypatch.setattr(config, "GROQ_BASE_URL", "u")
    assert config.get_llm_settings("  GROQ ")[0] == "m"


# ── Reproducibility ──────────────────────────────────────────────────────────

def test_set_reproducibility_updates_globals_and_rng():
    original = (config.SEED, config.TEMPERATURE)
    try:
        config.set_reproducibility(seed=123, temperature=0.0)
        assert config.SEED == 123
        assert config.TEMPERATURE == 0.0
        first = [random.random() for _ in range(3)]
        config.set_reproducibility(seed=123)
        second = [random.random() for _ in range(3)]
        assert first == second
    finally:
        config.set_reproducibility(seed=original[0], temperature=original[1])


def test_set_reproducibility_leaves_unspecified_values_alone():
    original = (config.SEED, config.TEMPERATURE)
    try:
        config.set_reproducibility(seed=7, temperature=0.3)
        config.set_reproducibility(seed=None, temperature=None)
        assert (config.SEED, config.TEMPERATURE) == (7, 0.3)
    finally:
        config.set_reproducibility(seed=original[0], temperature=original[1])


def test_run_metadata_has_provenance_fields():
    meta = config.get_run_metadata()
    assert {
        "timestamp",
        "seed",
        "temperature",
        "llm_backend",
        "model",
        "max_turns",
        "git_commit",
        "python_version",
    } <= set(meta)
    assert meta["seed"] == config.SEED
    assert meta["temperature"] == config.TEMPERATURE
    assert isinstance(meta["git_commit"], str)


# ── Defence registry / shim ──────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("spec", "expected_type", "expected_layers"),
    [
        ("0", type(None), 0),
        ("none", type(None), 0),
        ("1", defence.Layer1Defense, 1),
        ("layer2", defence.Layer2Defense, 1),
        ("l3", defence.Layer3Defense, 1),
        ("4", defence.Layer4Defense, 1),
        ("5", defence.Layer5Defense, 1),
    ],
)
def test_get_defense_single_layers(spec, expected_type, expected_layers):
    result = defence.get_defense(spec)
    assert isinstance(result, expected_type)
    if expected_layers:
        assert result.layer == int(spec.replace("layer", "").replace("l", ""))


@pytest.mark.parametrize(
    ("spec", "n_layers"),
    [
        ("1+2", 2),
        ("1+2+3", 3),
        ("1+2+3+4", 4),
        ("1+2+3+4+5", 5),
        ("full", 5),
        ("c4", 5),
        ("without_l1_l2", 2),
        ("3+5", 2),
    ],
)
def test_get_defense_pipelines(spec, n_layers):
    pipeline = defence.get_defense(spec)
    assert isinstance(pipeline, defence.DefensePipeline)
    assert len(pipeline.layers) == n_layers


def test_get_defense_rejects_unknown_spec():
    with pytest.raises(ValueError, match="Unknown defense specification"):
        defence.get_defense("nonsense")


def test_compatibility_shim_exports_same_classes():
    assert defense.Layer1Defense is defence.Layer1Defense
    assert defense.DefensePipeline is defence.DefensePipeline
    assert defense.get_defense is defence.get_defense
    assert set(defense.__all__) == set(defence.__all__)
