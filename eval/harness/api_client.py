#!/usr/bin/env python3
"""Thin OpenAI-compatible chat client used for the decoder/teacher API.

Reads the key from (in order): DEEPSEEK_API_KEY env var, an explicit --api
file, or api.txt. Also understands the `cot-decoder：sk-...` format.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

KEY_RE = re.compile(r"sk-[A-Za-z0-9_\-]+")

RECAST_ROOT = Path(__file__).resolve().parents[2]  # eval/harness -> RECAST
DEFAULT_API_FILE = RECAST_ROOT / "api.txt"


def load_api_key(api_file: str | Path | None = None,
                 env_var: str = "DEEPSEEK_API_KEY") -> str:
    env_key = os.environ.get(env_var, "").strip()
    if env_key:
        return env_key
    path = Path(api_file) if api_file else DEFAULT_API_FILE
    if not path.exists():
        sys.exit(f"[fatal] api file not found: {path}")
    match = KEY_RE.search(path.read_text(encoding="utf-8-sig"))
    if not match:
        sys.exit(f"[fatal] no `sk-...` key found in {path}")
    return match.group(0)


def mask_key(key: str) -> str:
    return key[:7] + "..." + key[-4:] if len(key) > 12 else "***"


class ChatClient:
    """Minimal wrapper around the OpenAI SDK with retries on transient errors."""

    def __init__(self, model: str, base_url: str | None = None,
                 api_file: str | Path | None = None, timeout: float = 240.0,
                 key_env: str = "DEEPSEEK_API_KEY"):
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("[fatal] openai package missing: pip install openai")
        self.model = model
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.key = load_api_key(api_file, env_var=key_env)
        self.client = OpenAI(api_key=self.key, base_url=self.base_url, timeout=timeout)

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.0,
             max_tokens: int = 512, retries: int = 4,
             json_mode: bool = False) -> dict[str, Any]:
        import time

        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = self.client.chat.completions.create(**kwargs)
                content = (resp.choices[0].message.content or "").strip()
                return {
                    "content": content,
                    "usage": getattr(resp, "usage", None),
                    "finish_reason": resp.choices[0].finish_reason,
                    "model": self.model,
                }
            except Exception as exc:  # noqa: BLE001 - surface API errors
                last_exc = exc
                status = getattr(exc, "status_code", None)
                if status in (401, 402, 403):
                    raise
                if attempt == retries:
                    break
                time.sleep(min(2 ** attempt, 20))
        assert last_exc is not None
        raise last_exc
