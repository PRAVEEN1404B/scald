"""Core tool-use agent loop for the PID project.

Backend-agnostic: works with Ollama (localhost:11434/v1), OpenAI, or any
OpenAI-compatible endpoint.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

import config as _config
from config import (
    LOG_DIR,
    MAX_TURNS,
    get_llm_settings,
    is_reasoning_model,
)

logger = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class ToolCallRecord:
    """One tool invocation record."""
    turn: int
    tool_name: str
    tool_input: dict[str, Any]
    tool_result: str


@dataclass
class AgentResult:
    """Complete result of an agent run."""
    final_text: str
    messages: list[dict[str, Any]]
    tool_trace: list[ToolCallRecord] = field(default_factory=list)
    injection_detected: bool = False          # Set if any detector flags injection
    turns_used: int = 0
    defense_name: str = "none"


# ── System Prompt ─────────────────────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided tools to help the user "
    "with their request. When you have finished, respond with a final answer."
)


# ── Agent Loop ────────────────────────────────────────────────────────────────

def run_agent(
    task: str,
    *,
    tools: list[dict] | None = None,
    tool_map: dict[str, Callable] | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    max_turns: int = MAX_TURNS,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    log_file: str | Path | None = None,
    tool_response_overrides: dict[str, str] | None = None,
    defense: Any = None,
) -> AgentResult:
    """Run the tool-use agent loop.

    Args:
        task: The user's task / instruction string.
        tools: OpenAI-format tool schema list. If None, no tools are offered.
        tool_map: Mapping of tool name -> callable((**kwargs) -> str).
        model: Model name to use.
        base_url: OpenAI-compatible API base URL.
        api_key: API key ("ollama" for local Ollama).
        max_turns: Maximum number of LLM calls before forced stop.
        system_prompt: System prompt string.
        log_file: Optional path to write turn-by-turn JSONL log.
        tool_response_overrides: Optional dict mapping tool_name -> fixed response string.
            When set, calling that tool returns the override instead of executing the real function.
            Used by InjecAgent adapter to inject pre-crafted responses.
        defense: Optional defense layer (e.g. 1, 'layer1', or BaseDefense instance).

    Returns:
        AgentResult with final text, full message history, tool trace, and defense status.
    """
    # Resolve LLM endpoint from LLM_BACKEND (Groq or Ollama) when not overridden.
    default_model, default_base_url, default_api_key = get_llm_settings()
    model = model or default_model
    base_url = base_url or default_base_url
    api_key = api_key or default_api_key
    client = OpenAI(base_url=base_url, api_key=api_key)

    # Resolve defense layer if specified
    from defence import get_defense
    defense_obj = get_defense(defense)

    if defense_obj is not None:
        system_prompt = defense_obj.protect_system_prompt(system_prompt)
        task = defense_obj.protect_user_prompt(task)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    tool_trace: list[ToolCallRecord] = []
    log_entries: list[dict] = []
    final_text = ""
    current_turn = 0

    for turn in range(1, max_turns + 1):
        current_turn = turn

        if defense_obj is not None:
            messages = defense_obj.on_turn_start(turn, messages, task)

        # ── LLM call ──
        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        # Reasoning models (gpt-oss, o-series) don't accept temperature
        if not is_reasoning_model(model):
            call_kwargs["temperature"] = _config.TEMPERATURE
        # Seed makes sampling reproducible on backends that support it.
        if _config.SEED is not None:
            call_kwargs["seed"] = _config.SEED
        if tools:
            call_kwargs["tools"] = tools
            call_kwargs["tool_choice"] = "auto"

        response = None
        for attempt in range(5):
            try:
                response = client.chat.completions.create(**call_kwargs)
                break
            except Exception as e:
                err_msg = str(e)
                # Some OpenAI-compatible servers reject the optional `seed` field.
                if "seed" in call_kwargs and "seed" in err_msg.lower():
                    logger.warning("Backend rejected `seed`; retrying without it")
                    call_kwargs.pop("seed", None)
                    continue
                logger.warning("LLM call failed on turn %d (attempt %d): %s", turn, attempt + 1, err_msg)
                if "429" in err_msg or "rate_limit" in err_msg.lower():
                    from config import GROQ_API_KEYS, rotate_groq_key
                    if len(GROQ_API_KEYS) > 1:
                        new_key = rotate_groq_key()
                        print(f"   Groq key rate limited. Rotating to key ...{new_key[-8:]}")
                        client = OpenAI(base_url=base_url, api_key=new_key)
                        time.sleep(1.0)
                        continue

                    wait_sec = 10.0
                    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", err_msg)
                    if m:
                        mins = float(m.group(1)) if m.group(1) else 0.0
                        secs = float(m.group(2)) if m.group(2) else 0.0
                        wait_sec = mins * 60 + secs + 2.0
                    print(f"   ⏳ Groq rate limit: waiting {int(wait_sec)}s for token refill...")
                    time.sleep(wait_sec)
                else:
                    time.sleep(1.0)
                if attempt == 4:
                    logger.error("LLM call failed after 5 attempts on turn %d: %s", turn, err_msg)
                    final_text = f"[Agent error on turn {turn}: {err_msg}]"
                    break

        if response is None:
            break

        choice = response.choices[0]
        assistant_msg = choice.message

        # ── Log the raw response ──
        log_entry = {
            "turn": turn,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": "assistant",
            "content": assistant_msg.content,
            "tool_calls": None,
            "finish_reason": choice.finish_reason,
        }

        # ── Handle tool calls ──
        if assistant_msg.tool_calls:
            # Build the assistant message dict for the conversation
            assistant_dict: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_msg.tool_calls
                ],
            }
            messages.append(assistant_dict)

            log_entry["tool_calls"] = []

            for tc in assistant_msg.tool_calls:
                func_name = tc.function.name
                try:
                    func_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    func_args = {"_raw": tc.function.arguments}

                # Defense tool call validation hook (for active policies)
                tool_blocked = False
                block_reason = None
                if defense_obj is not None:
                    allowed, block_reason = defense_obj.validate_tool_call(
                        func_name, func_args, context={"turn": turn, "messages": messages}
                    )
                    if not allowed:
                        tool_blocked = True

                if tool_blocked:
                    result_str = f"Blocked by defense policy: {block_reason}"
                elif tool_response_overrides and func_name in tool_response_overrides:
                    result_str = tool_response_overrides.pop(func_name)  # Use once
                elif tool_map and func_name in tool_map:
                    try:
                        result_str = str(tool_map[func_name](**func_args))
                    except Exception as e:
                        result_str = f"Tool error: {e}"
                else:
                    result_str = f"Error: Unknown tool '{func_name}'"

                # Record raw execution
                record = ToolCallRecord(
                    turn=turn,
                    tool_name=func_name,
                    tool_input=func_args,
                    tool_result=result_str,
                )
                tool_trace.append(record)

                log_entry["tool_calls"].append({
                    "tool_name": func_name,
                    "tool_input": func_args,
                    "tool_result": result_str[:500],  # Truncate for log readability
                })

                # Apply defense to tool result before feeding back to the model
                protected_result_str = result_str
                if defense_obj is not None and not tool_blocked:
                    protected_result_str = defense_obj.protect_tool_result(
                        tool_name=func_name,
                        result=result_str,
                        tool_args=func_args,
                        user_task=task,
                    )

                # Add tool result message
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": protected_result_str,
                })

            log_entries.append(log_entry)
            logger.info("Turn %d: %d tool call(s)", turn, len(assistant_msg.tool_calls))

        else:
            # ── Plain text response — we're done ──
            final_text = assistant_msg.content or ""
            messages.append({"role": "assistant", "content": final_text})
            log_entry["content"] = final_text
            log_entries.append(log_entry)
            logger.info("Turn %d: final text response", turn)
            break
    else:
        # max_turns exhausted
        final_text = final_text or "[Agent reached max turns without a final answer]"

    # ── Write log file ──
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            for entry in log_entries:
                f.write(json.dumps(entry, default=str) + "\n")

    return AgentResult(
        final_text=final_text,
        messages=messages,
        tool_trace=tool_trace,
        injection_detected=False,
        turns_used=current_turn,
        defense_name=defense_obj.name if defense_obj else "none",
    )


# ── Convenience: run with default tools ───────────────────────────────────────

def run_agent_with_default_tools(task: str, **kwargs) -> AgentResult:
    """Run the agent with the default read_file + send_email tools."""
    from tools import TOOL_MAP, TOOL_SCHEMAS
    return run_agent(task, tools=TOOL_SCHEMAS, tool_map=TOOL_MAP, **kwargs)


if __name__ == "__main__":
    # Quick manual test
    import sys
    logging.basicConfig(level=logging.INFO)
    task_str = " ".join(sys.argv[1:]) or "What is 2 + 2?"
    result = run_agent_with_default_tools(
        task_str,
        log_file=LOG_DIR / "manual_test.jsonl",
    )
    print(f"\n{'='*60}")
    print(f"Final answer: {result.final_text}")
    print(f"Tool calls: {len(result.tool_trace)}")
    for tc in result.tool_trace:
        print(f"  Turn {tc.turn}: {tc.tool_name}({tc.tool_input})")
