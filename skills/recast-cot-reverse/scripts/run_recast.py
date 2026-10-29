#!/usr/bin/env python3
"""Locate the RECAST repo and forward all arguments to recast_pipeline.py.

This keeps the skill portable: it works when invoked from any directory such
as a workspace root, a Claude Code project, or a Codex sandbox.

Search order:
  1. $RECAST_ROOT (either the RECAST repo or its parent)
  2. walk cwd and ancestors looking for recast_pipeline.py or recast_pipeline.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def find_repo() -> Path:
    env = os.environ.get("RECAST_ROOT", "").strip()
    if env:
        candidate = Path(env).expanduser().resolve()
        for path in (candidate / "recast_pipeline.py", candidate / "RECAST" / "recast_pipeline.py"):
            if path.is_file():
                return path.parent
        raise SystemExit(f"[fatal] RECAST_ROOT={candidate} does not contain recast_pipeline.py")

    cwd = Path.cwd().resolve()
    for base in (cwd, *cwd.parents):
        for path in (base / "recast_pipeline.py", base / "RECAST" / "recast_pipeline.py"):
            if path.is_file():
                return path.parent
    raise SystemExit(
        "[fatal] could not find recast_pipeline.py. Set RECAST_ROOT to the RECAST repo "
        "or run this from the project workspace.")


def main() -> int:
    repo = find_repo()
    cmd = [sys.executable, str(repo / "recast_pipeline.py"), *sys.argv[1:]]
    return subprocess.call(cmd, cwd=str(repo))


if __name__ == "__main__":
    raise SystemExit(main())
