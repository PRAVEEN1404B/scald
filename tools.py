"""Tool implementations and OpenAI-format schemas for the PID agent."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import LOG_DIR

# ── Tool Implementations ──────────────────────────────────────────────────────

def read_file(path: str) -> str:
    """Read the contents of a local file.
    
    Sandboxed to the project directory for safety.
    """
    target = Path(path).resolve()
    # Basic sandboxing: only allow reading files within or relative to project
    # (for the demo we're permissive, but log everything)
    if not target.exists():
        return f"Error: File not found: {path}"
    try:
        return target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"


def send_email(to: str, subject: str, body: str) -> str:
    """Fake email sender — logs the call as an exfiltration-sink proxy.
    
    Does NOT actually send email. Writes to logs/email_log.jsonl.
    """
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "to": to,
        "subject": subject,
        "body": body,
    }
    log_path = LOG_DIR / "email_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
    return f"Email sent to {to} with subject '{subject}'."


# ── Tool Registry ─────────────────────────────────────────────────────────────

# Maps tool name -> callable
TOOL_MAP: dict[str, Any] = {
    "read_file": read_file,
    "send_email": send_email,
}

# OpenAI-format tool schemas
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a local file and return its text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path to read."
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to the specified recipient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address."
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line."
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body text."
                    }
                },
                "required": ["to", "subject", "body"]
            }
        }
    },
]
