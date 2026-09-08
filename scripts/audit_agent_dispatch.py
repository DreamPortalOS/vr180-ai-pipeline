#!/usr/bin/env python3
"""W-3 (#306): stateless agent-dispatch audit — live git/gh, no ledger.

The previous version of this script audited ``docs/agent_tasks.json``, a
hand-maintained ledger of agent tasks.  That ledger was stale within twelve
hours of being introduced (#296): it still listed #291 as
``blocked-needs-tests`` and #292 as ``ready-for-review`` after both had
merged, and every recorded worktree path had been cleaned up.  Two sources of
truth that disagree are worse than one, and GitHub's issues / PRs / branches
are already authoritative.

So this script keeps **no state at all**.  Every run collects the current
picture from three live queries and reports on it:

1. ``git worktree list --porcelain``            — which worktrees exist, on
   what branch, at what HEAD
2. ``gh pr list --state open --json number,headRefName,mergeStateStatus``
3. ``gh issue list --state open --json number,title``

plus, per worktree, ``git status --porcelain`` (uncommitted work) and one
``git ls-remote --heads origin`` (which branches actually reached the remote).

Four alert classes — each one is a way an agent task silently stalls:

* **no open PR** — a worktree/branch exists but nothing is up for review
* **CI not green** — an open PR whose ``mergeStateStatus`` is not clean
* **uncommitted changes** — a worktree with work that exists nowhere else
* **branch not pushed** — a branch that only exists on this machine

``gh`` missing or unauthenticated is a hard failure (exit code 2), never a
quietly-empty report.  A silently-empty result is precisely the failure mode
that let a fabricated ``gh`` flag hide for a full release (#188); "no alerts"
must mean "GitHub says everything is fine", not "we could not ask".

Read-only: nothing here writes, pushes, comments, or edits.

Usage:
    python scripts/audit_agent_dispatch.py

Exit codes:
    0 — no alerts
    1 — at least one alert (see the report)
    2 — git/gh unusable (not installed, not authenticated, query failed)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Exit code for "the tools this audit depends on are unusable".
EXIT_TOOL_UNAVAILABLE = 2

#: ``mergeStateStatus`` values that mean GitHub is happy with the PR.
#: ``HAS_HOOKS`` is clean-plus-pre-receive-hooks, i.e. also mergeable.
GREEN_MERGE_STATES = frozenset({"CLEAN", "HAS_HOOKS"})

#: Plain-English gloss for the non-green ``mergeStateStatus`` values, so the
#: report says what to *do* rather than just echoing a GitHub enum.
MERGE_STATE_HINT = {
    "BEHIND": "branch is behind its base — update the branch",
    "BLOCKED": "a required check or review is not satisfied",
    "DIRTY": "merge conflicts with the base branch",
    "DRAFT": "draft PR — mark it ready before CI gates it",
    "UNKNOWN": "GitHub has not finished computing mergeability — re-run",
    "UNSTABLE": "a non-required check is failing",
}


class AuditToolError(RuntimeError):
    """A tool this audit depends on (``git`` / ``gh``) could not be used.

    Raised for a missing binary, a failed authentication check, a non-zero
    query, or unparseable output.  Callers map this to exit code 2 — it is
    never downgraded into an empty report.
    """


# ---------------------------------------------------------------------------
# Live data sources (injectable runner for tests)
# ---------------------------------------------------------------------------


def _invoke(
    cmd: list[str],
    *,
    cwd: Path = REPO_ROOT,
    runner: Any = subprocess.run,
    missing_hint: str = "",
) -> Any:
    """Execute ``cmd`` (list form, never ``shell=True``); return the result.

    Only a *missing binary* raises here — a non-zero exit is left to the
    caller, because ``gh auth status`` reports auth state through its exit
    code and must be inspected rather than treated as a crash.

    Tests inject ``runner``; no real ``git`` / ``gh`` process is spawned in CI.
    """
    try:
        return runner(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=60, check=False)
    except OSError as exc:
        raise AuditToolError(
            f"cannot execute {cmd[0]!r} ({exc}). This audit reads live state from git and "
            f"GitHub; there is no offline fallback. {missing_hint or 'Install it and re-run.'}"
        ) from exc


def _run(cmd: list[str], *, what: str, cwd: Path = REPO_ROOT, runner: Any = subprocess.run) -> str:
    """Run ``cmd`` and return stdout, raising on any non-zero exit.

    ``what`` names the query in error messages so a failure points at the
    thing that broke rather than at this helper.
    """
    proc = _invoke(cmd, cwd=cwd, runner=runner)
    if proc.returncode != 0:
        stderr = (getattr(proc, "stderr", "") or "").strip()
        raise AuditToolError(f"{what}: `{' '.join(cmd)}` failed (rc={proc.returncode}): {stderr or '<no stderr>'}")
    return getattr(proc, "stdout", "") or ""


def require_gh(runner: Any = subprocess.run) -> None:
    """Fail loudly (exit 2 via :class:`AuditToolError`) unless ``gh`` is usable.

    Checked up front so the report never starts printing a half-picture.  The
    two failures get two messages because the fixes differ: ``gh`` not on PATH
    (install it) versus ``gh`` present but logged out (``gh auth login``).
    ``gh auth status`` exits non-zero when no account is authenticated, and
    writes its diagnosis to stderr on some versions and stdout on others, so
    the decision is made on the exit code and both streams are quoted back.
    """
    proc = _invoke(
        ["gh", "auth", "status"],
        runner=runner,
        missing_hint="Install GitHub CLI (https://cli.github.com/) and run `gh auth login`.",
    )
    if proc.returncode != 0:
        detail = " ".join(
            part.strip()
            for part in ((getattr(proc, "stderr", "") or ""), (getattr(proc, "stdout", "") or ""))
            if part.strip()
        )
        raise AuditToolError(
            f"gh CLI is not authenticated (`gh auth status` exited {proc.returncode}). "
            "Run `gh auth login`, then re-run this audit. Refusing to emit an empty report: "
            "'no alerts' must mean GitHub said so, not that we could not ask. "
            f"gh said: {detail or '<no output>'}"
        )


def _gh_json(argv: list[str], *, what: str, runner: Any = subprocess.run) -> list[dict[str, Any]]:
    """Run a ``gh ... --json`` query and return the parsed list."""
    stdout = _run(["gh", *argv], what=what, runner=runner)
    try:
        parsed = json.loads(stdout or "[]")
    except json.JSONDecodeError as exc:
        raise AuditToolError(f"{what}: gh returned unparseable JSON ({exc}): {stdout[:200]!r}") from exc
    if not isinstance(parsed, list):
        raise AuditToolError(f"{what}: expected a JSON list from gh, got {type(parsed).__name__}")
    return parsed


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Worktree:
    """One entry from ``git worktree list --porcelain``."""

    path: str
    branch: str = ""  # "" for a detached HEAD
    head: str = ""
    dirty: str = ""  # human summary of `git status --porcelain`; "" when clean
    exists: bool = True

    @property
    def label(self) -> str:
        return f"{self.path} [{self.branch or 'detached'} @ {self.head[:7] or '???????'}]"


@dataclass
class PullRequest:
    """One entry from ``gh pr list --state open``."""

    number: int
    branch: str
    merge_state: str

    @property
    def is_green(self) -> bool:
        return self.merge_state.upper() in GREEN_MERGE_STATES


@dataclass
class Snapshot:
    """Everything one audit run collected. Built fresh each time; never saved."""

    worktrees: list[Worktree] = field(default_factory=list)
    pull_requests: list[PullRequest] = field(default_factory=list)
    open_issues: list[dict[str, Any]] = field(default_factory=list)
    remote_branches: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def parse_worktrees(porcelain: str) -> list[Worktree]:
    """Parse ``git worktree list --porcelain`` into :class:`Worktree` records.

    The porcelain format is one blank-line-separated block per worktree, with
    a leading ``worktree <path>`` line and optional ``HEAD``/``branch``/
    ``detached``/``bare`` attribute lines.
    """
    found: list[Worktree] = []
    for line in porcelain.splitlines():
        if line.startswith("worktree "):
            found.append(Worktree(path=line.removeprefix("worktree ").replace("\\", "/")))
        elif not found:
            continue
        elif line.startswith("HEAD "):
            found[-1].head = line.removeprefix("HEAD ").strip()
        elif line.startswith("branch "):
            found[-1].branch = line.removeprefix("branch ").removeprefix("refs/heads/").strip()
    return found


def _summarize_status(porcelain: str) -> str:
    """Summarize ``git status --porcelain`` as e.g. ``2 modified, 1 untracked``.

    Untracked files are counted separately because they mean something
    different from unstaged edits: a file that exists on exactly one machine
    and in no commit anywhere.
    """
    tracked = 0
    untracked = 0
    for line in porcelain.splitlines():
        if not line.strip():
            continue
        if line.startswith("??"):
            untracked += 1
        else:
            tracked += 1
    parts = []
    if tracked:
        parts.append(f"{tracked} modified")
    if untracked:
        parts.append(f"{untracked} untracked")
    return ", ".join(parts)


def _parse_remote_branches(ls_remote: str) -> set[str]:
    """Parse ``git ls-remote --heads origin`` into a set of branch names."""
    branches: set[str] = set()
    for line in ls_remote.splitlines():
        _, _, ref = line.partition("\t")
        ref = ref.strip()
        if ref.startswith("refs/heads/"):
            branches.add(ref.removeprefix("refs/heads/"))
    return branches


def collect(runner: Any = subprocess.run) -> Snapshot:
    """Collect the live picture. Every field comes from a fresh query."""
    snapshot = Snapshot()

    snapshot.worktrees = parse_worktrees(
        _run(["git", "worktree", "list", "--porcelain"], what="worktree listing", runner=runner)
    )
    for worktree in snapshot.worktrees:
        path = Path(worktree.path)
        if not path.is_dir():
            # A registered-but-deleted worktree: exactly the rot that made the
            # hand-kept ledger useless. Reported, but not status-checked.
            worktree.exists = False
            continue
        worktree.dirty = _summarize_status(
            _run(
                ["git", "status", "--porcelain"],
                what=f"status of {worktree.path}",
                cwd=path,
                runner=runner,
            )
        )

    snapshot.remote_branches = _parse_remote_branches(
        _run(["git", "ls-remote", "--heads", "origin"], what="remote branch listing", runner=runner)
    )

    snapshot.pull_requests = [
        PullRequest(
            number=int(item.get("number", 0)),
            branch=str(item.get("headRefName", "")),
            merge_state=str(item.get("mergeStateStatus", "UNKNOWN")),
        )
        for item in _gh_json(
            ["pr", "list", "--state", "open", "--json", "number,headRefName,mergeStateStatus"],
            what="open PR query",
            runner=runner,
        )
    ]
    snapshot.open_issues = _gh_json(
        ["issue", "list", "--state", "open", "--json", "number,title"],
        what="open issue query",
        runner=runner,
    )
    return snapshot


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report(snapshot: Snapshot, stream: Any = sys.stdout) -> int:
    """Print the audit report; return the number of alerts raised."""
    pr_by_branch = {pr.branch: pr for pr in snapshot.pull_requests if pr.branch}
    alerts = 0

    print("=== worktrees (git worktree list --porcelain) ===", file=stream)
    if not snapshot.worktrees:
        print("  (none)", file=stream)
    for worktree in snapshot.worktrees:
        problems: list[str] = []
        notes: list[str] = []
        if not worktree.exists:
            notes.append("path is gone — run `git worktree prune`")
        elif worktree.dirty:
            problems.append(f"uncommitted changes ({worktree.dirty})")
        if worktree.branch:
            if worktree.branch not in snapshot.remote_branches:
                problems.append("branch not pushed to origin")
            if worktree.branch not in pr_by_branch:
                problems.append("no open PR for this branch")
            else:
                notes.append(f"PR #{pr_by_branch[worktree.branch].number}")
        else:
            notes.append("detached HEAD — no branch to track")
        marker = "ALERT" if problems else "OK"
        detail = "; ".join(problems + notes) or "clean, pushed, PR open"
        print(f"[{marker}] {worktree.label}: {detail}", file=stream)
        alerts += len(problems)

    print("", file=stream)
    print("=== open PRs (gh pr list --state open) ===", file=stream)
    if not snapshot.pull_requests:
        print("  (none)", file=stream)
    for pr in sorted(snapshot.pull_requests, key=lambda item: item.number):
        if pr.is_green:
            print(f"[OK]    PR #{pr.number} ({pr.branch}): {pr.merge_state}", file=stream)
            continue
        hint = MERGE_STATE_HINT.get(pr.merge_state.upper(), "not mergeable")
        print(
            f"[ALERT] PR #{pr.number} ({pr.branch}): CI not green — {pr.merge_state} ({hint})",
            file=stream,
        )
        alerts += 1

    print("", file=stream)
    print("=== open issues (gh issue list --state open) ===", file=stream)
    active_branches = {wt.branch for wt in snapshot.worktrees if wt.branch} | set(pr_by_branch)
    if not snapshot.open_issues:
        print("  (none)", file=stream)
    for issue in snapshot.open_issues:
        number = issue.get("number")
        in_flight = any(f"issue-{number}" in branch for branch in active_branches)
        flag = "in flight" if in_flight else "no branch/PR yet"
        print(f"[INFO]  #{number} ({flag}): {issue.get('title', '')}", file=stream)

    print("", file=stream)
    print(
        f"Summary: {alerts} alert(s) across {len(snapshot.worktrees)} worktree(s), "
        f"{len(snapshot.pull_requests)} open PR(s), {len(snapshot.open_issues)} open issue(s).",
        file=stream,
    )
    return alerts


def audit(runner: Any = subprocess.run, stream: Any = sys.stdout) -> int:
    """Run one full audit. Returns the process exit code (0 / 1)."""
    require_gh(runner=runner)
    return 1 if report(collect(runner=runner), stream=stream) else 0


def main(argv: list[str] | None = None, runner: Any = subprocess.run, stream: Any = None) -> int:
    """CLI entry point. ``runner`` / ``stream`` exist so tests can drive it."""
    parser = argparse.ArgumentParser(
        description=(
            "Stateless agent-dispatch audit: reports worktrees without an open PR, "
            "open PRs whose CI is not green, worktrees with uncommitted changes, and "
            "branches that were never pushed. Reads live git/gh state; keeps no ledger."
        )
    )
    parser.parse_args(argv)
    try:
        return audit(runner=runner, stream=sys.stdout if stream is None else stream)
    except AuditToolError as exc:
        # Exit 2, never a quiet empty report: the operator must be able to tell
        # "GitHub says all clear" apart from "we never reached GitHub".
        print(f"[audit_agent_dispatch] {exc}", file=sys.stderr)
        return EXIT_TOOL_UNAVAILABLE


if __name__ == "__main__":
    raise SystemExit(main())
