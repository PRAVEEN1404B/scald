"""Centralised configuration for the PID project."""
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).parent

# ── LLM Backends ──
# Local Ollama (fallback)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")

# Cloud Groq (Free & Blazing Fast — 0% local Mac CPU/RAM usage)
# Get a free key at: https://console.groq.com/keys
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

_groq_keys_raw = os.getenv("GROQ_API_KEYS") or os.getenv("GROQ_API_KEY", "")
GROQ_API_KEYS = [k.strip() for k in _groq_keys_raw.split(",") if k.strip()]
_current_groq_key_idx = 0

def get_current_groq_key() -> str:
    global _current_groq_key_idx
    if not GROQ_API_KEYS:
        return os.getenv("GROQ_API_KEY", "")
    return GROQ_API_KEYS[_current_groq_key_idx % len(GROQ_API_KEYS)]

def rotate_groq_key() -> str:
    global _current_groq_key_idx
    if len(GROQ_API_KEYS) <= 1:
        return get_current_groq_key()
    _current_groq_key_idx = (_current_groq_key_idx + 1) % len(GROQ_API_KEYS)
    return GROQ_API_KEYS[_current_groq_key_idx]

GROQ_API_KEY = get_current_groq_key()

# Active Backend: automatically use Groq if GROQ_API_KEY is set, else Ollama
LLM_BACKEND = os.getenv("LLM_BACKEND", "groq" if GROQ_API_KEY else "ollama")

# Causal-probe client settings. Groq/Qwen responses often take >3s; the old
# 2.5s timeout caused silent heuristic fallback, so probes never really ran.
PROBE_TIMEOUT = float(os.getenv("PROBE_TIMEOUT", "20"))
PROBE_CONNECT_TIMEOUT = float(os.getenv("PROBE_CONNECT_TIMEOUT", "5"))
PROBE_MAX_RETRIES = int(os.getenv("PROBE_MAX_RETRIES", "1"))

# Reasoning models (gpt-oss, o-series) spend tokens on hidden reasoning before
# emitting YES/NO. Give them enough budget and keep reasoning effort low.
PROBE_MAX_TOKENS = int(os.getenv("PROBE_MAX_TOKENS", "64"))
PROBE_REASONING_EFFORT = os.getenv("PROBE_REASONING_EFFORT", "low")


def is_reasoning_model(model: str) -> bool:
    """True for models that emit hidden reasoning tokens before the answer."""
    name = (model or "").lower()
    return any(tag in name for tag in ("gpt-oss", "o1", "o3", "o4", "reasoner", "deepseek-r1"))


def get_llm_settings(backend: str | None = None) -> tuple[str, str, str]:
    """Resolve ``(model, base_url, api_key)`` for the active LLM backend.

    Single source of truth for the agent loop and every defence layer, so
    that ``LLM_BACKEND``/``GROQ_*`` in ``.env`` take effect everywhere
    instead of silently falling back to local Ollama.
    """
    active = (backend or LLM_BACKEND or "ollama").strip().strip("'\"").lower()
    current_key = get_current_groq_key()
    if active == "groq" and current_key:
        return GROQ_MODEL, GROQ_BASE_URL, current_key
    return OLLAMA_MODEL, OLLAMA_BASE_URL, OLLAMA_API_KEY


ANTHROPIC_MODEL = "claude-sonnet-4-20250514"  # For final citable runs

# ── Agent Settings ──
MAX_TURNS = 15

# ── Reproducibility ──
# Same seed + temperature + model + dataset should reproduce the same run.
# Override per-run with `--seed` / `--temperature` on the CLI scripts, or via env.
SEED = int(os.getenv("PID_SEED", "0"))
TEMPERATURE = float(os.getenv("PID_TEMPERATURE", "0.1"))


def set_reproducibility(seed: int | None = None, temperature: float | None = None) -> None:
    """Apply global seed/temperature overrides for this process.

    Seeds Python's ``random`` module (used by the harness, the case generator,
    and the trivial/random baselines) and stores the values on this module so
    that ``agent.py`` sends them to the LLM backend on every call.
    """
    global SEED, TEMPERATURE
    if seed is not None:
        SEED = int(seed)
    if temperature is not None:
        TEMPERATURE = float(temperature)
    random.seed(SEED)


def add_reproducibility_args(parser) -> None:
    """Attach the standard ``--seed`` / ``--temperature`` flags to an argparse parser."""
    group = parser.add_argument_group("reproducibility")
    group.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Random/LLM seed for reproducible runs (default: {SEED} or PID_SEED)",
    )
    group.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=f"LLM sampling temperature (default: {TEMPERATURE} or PID_TEMPERATURE)",
    )


def apply_reproducibility_args(args) -> None:
    """Read ``--seed`` / ``--temperature`` off a parsed argparse namespace."""
    set_reproducibility(
        seed=getattr(args, "seed", None),
        temperature=getattr(args, "temperature", None),
    )


def _git_commit() -> str:
    """Short commit hash of this project, or a clear marker if it is not a git repo."""
    if not (PROJECT_ROOT / ".git").exists():
        return "unknown (not a git repository)"
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def get_run_metadata() -> dict:
    """Everything needed to attribute a result file to a specific configuration."""
    model, _, _ = get_llm_settings()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "temperature": TEMPERATURE,
        "llm_backend": LLM_BACKEND,
        "model": model,
        "max_turns": MAX_TURNS,
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
    }

# ── Paths ──
LOG_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"
DEMO_DIR = PROJECT_ROOT / "demo"

# Ensure directories exist
for d in [LOG_DIR, RESULTS_DIR, DEMO_DIR]:
    d.mkdir(exist_ok=True)

