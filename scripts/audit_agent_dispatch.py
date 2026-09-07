"""Audit Claude/Cursor task ledger against Git worktrees."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "docs" / "agent_tasks.json"


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _worktrees() -> dict[str, dict[str, str]]:
    result = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    found: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            current = {"path": line.removeprefix("worktree ")}
            found[current["path"].replace("\\", "/")] = current
        elif current is not None and line.startswith("HEAD "):
            current["head"] = line.removeprefix("HEAD ")
        elif current is not None and line.startswith("branch "):
            current["branch"] = line.removeprefix("branch refs/heads/")
    return found


def _status(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    return "dirty" if result.stdout.strip() else "clean"


def audit(ledger_path: Path, stale_hours: float) -> int:
    data = json.loads(ledger_path.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc)
    worktrees = _worktrees()
    failures: list[str] = []

    for task in data["tasks"]:
        task_id = task["id"]
        raw_path = task["worktree"]
        path = Path(raw_path)
        path_key = raw_path.replace("\\", "/")
        entry = worktrees.get(path_key)
        problems: list[str] = []
        if entry is None or not path.exists():
            if task["status"] != "done":
                problems.append("worktree missing")
        else:
            if entry.get("branch") != task["branch"]:
                problems.append(f"branch={entry.get('branch')!r}")
            if not entry.get("head", "").startswith(task["head"]):
                problems.append(f"head={entry.get('head', '')[:12]}")
            if _status(path) == "dirty" and task["status"] in {"ready-for-review", "ready-to-merge", "done"}:
                problems.append("unexpected dirty worktree")
        age = (now - _parse_time(task["last_checked_at"])).total_seconds() / 3600
        if age > stale_hours and task["status"] not in {"done", "stopped"}:
            problems.append(f"stale heartbeat {age:.1f}h")
        marker = "OK" if not problems else "ALERT"
        print(f"[{marker}] #{task_id} {task['status']} {task['branch']}: {', '.join(problems) or 'tracked'}")
        if problems:
            failures.append(task_id)

    if failures:
        print("Audit failed for: " + ", ".join(f"#{task_id}" for task_id in failures), file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the Claude/Cursor task ledger against Git worktrees.")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--stale-hours", type=float, default=4.0)
    args = parser.parse_args()
    return audit(args.ledger, args.stale_hours)


if __name__ == "__main__":
    raise SystemExit(main())
