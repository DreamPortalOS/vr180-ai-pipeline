"""W-3 (#306): tests for the stateless agent-dispatch audit.

``scripts/audit_agent_dispatch.py`` used to compare a hand-maintained ledger
(``docs/agent_tasks.json``) against local worktrees.  The ledger went stale
within twelve hours of being introduced, so the script now keeps no state and
asks ``git`` / ``gh`` directly on every run.

Every ``git`` and ``gh`` call here is injected through the module's ``runner``
seam (:class:`FakeTools`) — CI is CPU-only, offline, and has no GitHub auth,
so no real process is ever spawned.  :class:`FakeTools` also records the argv
of every call, which lets the tests assert the *shape* of the invocations
(list form, no ``shell=True``) alongside the report they produce.

Covered:

* one test per alert class — no open PR / CI not green / uncommitted changes /
  branch not pushed
* ``gh`` missing from PATH and ``gh`` present-but-logged-out → exit code 2 with
  a readable message (never a silently empty report)
* a failed ``gh`` query → exit 2 for the same reason
* the all-clear path → exit 0
* the ledger file is really gone and nothing re-creates it
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from scripts.audit_agent_dispatch import (
    EXIT_TOOL_UNAVAILABLE,
    AuditToolError,
    Snapshot,
    Worktree,
    audit,
    main,
    parse_worktrees,
    report,
    require_gh,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# argv prefixes the audit is contractually required to issue.
GIT_WORKTREES = ("git", "worktree", "list", "--porcelain")
GIT_STATUS = ("git", "status", "--porcelain")
GIT_LS_REMOTE = ("git", "ls-remote", "--heads", "origin")
GH_AUTH = ("gh", "auth", "status")
GH_PR_LIST = ("gh", "pr", "list")
GH_ISSUE_LIST = ("gh", "issue", "list")


# ---------------------------------------------------------------------------
# Fake git / gh
# ---------------------------------------------------------------------------


class FakeTools:
    """A stand-in for ``subprocess.run`` covering every git/gh call the audit makes.

    Constructed from the *live state* a scenario wants to simulate (which
    worktrees exist, which branches reached origin, which PRs / issues are
    open) rather than from canned stdout, so each test reads as a situation
    rather than as a pile of strings.
    """

    def __init__(
        self,
        *,
        worktrees: list[dict[str, str]] | None = None,
        status_by_path: dict[str, str] | None = None,
        remote_branches: list[str] | None = None,
        prs: list[dict[str, Any]] | None = None,
        issues: list[dict[str, Any]] | None = None,
        gh_auth_rc: int = 0,
        gh_auth_stderr: str = "",
        missing_binaries: tuple[str, ...] = (),
        failing_prefix: tuple[str, ...] | None = None,
    ) -> None:
        self.worktrees = worktrees or []
        self.status_by_path = {Path(k).as_posix(): v for k, v in (status_by_path or {}).items()}
        self.remote_branches = remote_branches or []
        self.prs = prs or []
        self.issues = issues if issues is not None else []
        self.gh_auth_rc = gh_auth_rc
        self.gh_auth_stderr = gh_auth_stderr
        self.missing_binaries = missing_binaries
        self.failing_prefix = failing_prefix
        self.calls: list[list[str]] = []

    # -- helpers ---------------------------------------------------------
    def _worktree_porcelain(self) -> str:
        blocks = []
        for entry in self.worktrees:
            lines = [f"worktree {entry['path']}", f"HEAD {entry.get('head', 'deadbeef' * 5)}"]
            if entry.get("branch"):
                lines.append(f"branch refs/heads/{entry['branch']}")
            else:
                lines.append("detached")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks) + ("\n" if blocks else "")

    def _ls_remote(self) -> str:
        return "".join(f"{'a' * 40}\trefs/heads/{branch}\n" for branch in self.remote_branches)

    # -- the runner seam -------------------------------------------------
    def __call__(self, cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        # Discipline check: list form, never a shell string, never check=True.
        assert isinstance(cmd, list) and all(isinstance(part, str) for part in cmd), cmd
        assert "shell" not in kwargs, "subprocess must never be invoked with shell=True"
        assert kwargs.get("check") is False, "the audit inspects return codes itself"
        self.calls.append(list(cmd))

        if cmd[0] in self.missing_binaries:
            raise FileNotFoundError(2, "No such file or directory", cmd[0])
        argv = tuple(cmd)
        if self.failing_prefix and argv[: len(self.failing_prefix)] == self.failing_prefix:
            return SimpleNamespace(returncode=1, stdout="", stderr="boom")

        if argv[: len(GH_AUTH)] == GH_AUTH:
            return SimpleNamespace(returncode=self.gh_auth_rc, stdout="", stderr=self.gh_auth_stderr)
        if argv[: len(GIT_WORKTREES)] == GIT_WORKTREES:
            return SimpleNamespace(returncode=0, stdout=self._worktree_porcelain(), stderr="")
        if argv[: len(GIT_STATUS)] == GIT_STATUS:
            key = Path(kwargs["cwd"]).as_posix()
            return SimpleNamespace(returncode=0, stdout=self.status_by_path.get(key, ""), stderr="")
        if argv[: len(GIT_LS_REMOTE)] == GIT_LS_REMOTE:
            return SimpleNamespace(returncode=0, stdout=self._ls_remote(), stderr="")
        if argv[: len(GH_PR_LIST)] == GH_PR_LIST:
            return SimpleNamespace(returncode=0, stdout=json.dumps(self.prs), stderr="")
        if argv[: len(GH_ISSUE_LIST)] == GH_ISSUE_LIST:
            return SimpleNamespace(returncode=0, stdout=json.dumps(self.issues), stderr="")
        raise AssertionError(f"unstubbed command: {cmd}")


@pytest.fixture
def worktree_dir(tmp_path: Path) -> Path:
    """A real directory standing in for an agent worktree.

    It must exist on disk because ``collect()`` skips the ``git status`` probe
    for registered-but-deleted worktrees — the exact rot that made the ledger
    useless.
    """
    path = tmp_path / "agent-worktree"
    path.mkdir()
    return path


def _healthy(worktree_dir: Path, **overrides: Any) -> FakeTools:
    """A scenario where everything is fine; each test breaks exactly one thing."""
    defaults: dict[str, Any] = {
        "worktrees": [{"path": worktree_dir.as_posix(), "branch": "feat/issue-306-x", "head": "abc1234def"}],
        "status_by_path": {},
        "remote_branches": ["main", "feat/issue-306-x"],
        "prs": [{"number": 500, "headRefName": "feat/issue-306-x", "mergeStateStatus": "CLEAN"}],
        "issues": [{"number": 306, "title": "W-3: stateless audit"}],
    }
    defaults.update(overrides)
    return FakeTools(**defaults)


def _run_audit(tools: FakeTools) -> tuple[int, str]:
    stream = io.StringIO()
    code = audit(runner=tools, stream=stream)
    return code, stream.getvalue()


# ---------------------------------------------------------------------------
# Baseline: the all-clear path
# ---------------------------------------------------------------------------


def test_healthy_state_exits_zero_with_no_alerts(worktree_dir: Path) -> None:
    """Clean worktree + pushed branch + green open PR ⇒ exit 0, zero alerts."""
    code, out = _run_audit(_healthy(worktree_dir))
    assert code == 0, out
    assert "[ALERT]" not in out
    assert "PR #500" in out
    assert "Summary: 0 alert(s)" in out


def test_audit_issues_exactly_the_documented_live_queries(worktree_dir: Path) -> None:
    """The audit's whole contract is *which* live queries it runs — pin them.

    If a query is dropped the report silently narrows, which is the failure
    mode #306 exists to kill.
    """
    tools = _healthy(worktree_dir)
    _run_audit(tools)
    issued = [tuple(cmd) for cmd in tools.calls]
    assert GH_AUTH in issued
    assert GIT_WORKTREES in issued
    assert GIT_STATUS in issued
    assert GIT_LS_REMOTE in issued
    assert (
        "gh",
        "pr",
        "list",
        "--state",
        "open",
        "--json",
        "number,headRefName,mergeStateStatus",
    ) in issued
    assert ("gh", "issue", "list", "--state", "open", "--json", "number,title") in issued


# ---------------------------------------------------------------------------
# Alert class 1/4: worktree branch with no open PR
# ---------------------------------------------------------------------------


def test_alert_worktree_branch_without_open_pr(worktree_dir: Path) -> None:
    """A branch someone is working on but never opened a PR for is a stall."""
    code, out = _run_audit(_healthy(worktree_dir, prs=[]))
    assert code == 1
    assert "no open PR for this branch" in out
    assert "[ALERT]" in out
    assert "Summary: 1 alert(s)" in out


# ---------------------------------------------------------------------------
# Alert class 2/4: open PR whose CI is not green
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("merge_state", "hint_fragment"),
    [
        ("BLOCKED", "required check"),
        ("DIRTY", "merge conflicts"),
        ("UNSTABLE", "non-required check"),
        ("WEIRD_NEW_ENUM", "not mergeable"),
    ],
)
def test_alert_open_pr_with_non_green_ci(worktree_dir: Path, merge_state: str, hint_fragment: str) -> None:
    """Any non-CLEAN ``mergeStateStatus`` is reported, unknown enums included.

    An unrecognised value must still alert rather than fall through as "green"
    — GitHub adds enum members, and defaulting to OK would hide broken PRs.
    """
    tools = _healthy(
        worktree_dir,
        prs=[{"number": 500, "headRefName": "feat/issue-306-x", "mergeStateStatus": merge_state}],
    )
    code, out = _run_audit(tools)
    assert code == 1
    assert f"PR #500 (feat/issue-306-x): CI not green — {merge_state}" in out
    assert hint_fragment in out


def test_has_hooks_counts_as_green(worktree_dir: Path) -> None:
    """``HAS_HOOKS`` is CLEAN-plus-pre-receive-hooks — mergeable, not an alert."""
    tools = _healthy(
        worktree_dir,
        prs=[{"number": 500, "headRefName": "feat/issue-306-x", "mergeStateStatus": "HAS_HOOKS"}],
    )
    code, out = _run_audit(tools)
    assert code == 0, out
    assert "[ALERT]" not in out


# ---------------------------------------------------------------------------
# Alert class 3/4: worktree with uncommitted changes
# ---------------------------------------------------------------------------


def test_alert_worktree_with_uncommitted_changes(worktree_dir: Path) -> None:
    """Modified and untracked files are counted separately in the message."""
    tools = _healthy(
        worktree_dir,
        status_by_path={worktree_dir: " M scripts/a.py\nA  tests/b.py\n?? scratch.txt\n"},
    )
    code, out = _run_audit(tools)
    assert code == 1
    assert "uncommitted changes (2 modified, 1 untracked)" in out


def test_registered_but_deleted_worktree_is_reported_not_probed(tmp_path: Path) -> None:
    """A worktree whose directory is gone must not crash the ``git status`` probe.

    This is the concrete rot from #296: the ledger's recorded worktree paths
    had all been cleaned up. The audit reports it (with the prune hint) and
    skips the status call instead of raising.
    """
    ghost = tmp_path / "already-removed"
    tools = _healthy(
        tmp_path / "unused",
        worktrees=[{"path": ghost.as_posix(), "branch": "feat/gone", "head": "abc1234"}],
        remote_branches=["main", "feat/gone"],
        prs=[{"number": 501, "headRefName": "feat/gone", "mergeStateStatus": "CLEAN"}],
    )
    code, out = _run_audit(tools)
    assert "git worktree prune" in out
    assert code == 0, out
    assert not any(tuple(cmd)[:3] == GIT_STATUS for cmd in tools.calls)


# ---------------------------------------------------------------------------
# Alert class 4/4: branch never pushed
# ---------------------------------------------------------------------------


def test_alert_branch_not_pushed_to_origin(worktree_dir: Path) -> None:
    """A branch missing from ``git ls-remote --heads origin`` only exists here."""
    tools = _healthy(worktree_dir, remote_branches=["main"], prs=[])
    code, out = _run_audit(tools)
    assert code == 1
    assert "branch not pushed to origin" in out


def test_detached_head_worktree_is_informational(worktree_dir: Path) -> None:
    """A detached worktree has no branch to push or open a PR for — no alert."""
    tools = _healthy(
        worktree_dir,
        worktrees=[{"path": worktree_dir.as_posix(), "head": "abc1234def"}],
        prs=[],
    )
    code, out = _run_audit(tools)
    assert code == 0, out
    assert "detached HEAD" in out


# ---------------------------------------------------------------------------
# gh unavailable → exit 2, never a silently empty report
# ---------------------------------------------------------------------------


def test_missing_gh_binary_exits_two(worktree_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``gh`` not on PATH is a hard failure with an actionable message."""
    tools = _healthy(worktree_dir, missing_binaries=("gh",))
    stream = io.StringIO()
    code = main(argv=[], runner=tools, stream=stream)
    assert code == EXIT_TOOL_UNAVAILABLE
    err = capsys.readouterr().err
    assert "gh" in err
    assert "gh auth login" in err
    assert "cli.github.com" in err
    # Nothing was reported: the audit refuses to print a partial picture.
    assert stream.getvalue() == ""


def test_logged_out_gh_exits_two(worktree_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``gh`` installed but logged out ⇒ exit 2, quoting gh's own diagnosis."""
    tools = _healthy(worktree_dir, gh_auth_rc=1, gh_auth_stderr="You are not logged into any GitHub hosts.")
    stream = io.StringIO()
    code = main(argv=[], runner=tools, stream=stream)
    assert code == EXIT_TOOL_UNAVAILABLE
    err = capsys.readouterr().err
    assert "not authenticated" in err
    assert "gh auth login" in err
    assert "You are not logged into any GitHub hosts." in err
    assert stream.getvalue() == ""


def test_failed_gh_query_exits_two_instead_of_empty_report(worktree_dir: Path, capsys) -> None:
    """A gh query that fails mid-run must not degrade into "no open PRs".

    Returning ``[]`` here would read as "nothing in flight" — the #188 failure
    mode where a rejected flag looked like an empty result.
    """
    tools = _healthy(worktree_dir, failing_prefix=GH_PR_LIST)
    code = main(argv=[], runner=tools, stream=io.StringIO())
    assert code == EXIT_TOOL_UNAVAILABLE
    assert "open PR query" in capsys.readouterr().err


def test_require_gh_accepts_authenticated_cli(worktree_dir: Path) -> None:
    """The happy path returns quietly (and is what the audit gates on)."""
    require_gh(runner=_healthy(worktree_dir))


def test_missing_git_binary_also_exits_two(worktree_dir: Path, capsys) -> None:
    """git missing is the same class of failure as gh missing: exit 2."""
    tools = _healthy(worktree_dir, missing_binaries=("git",))
    code = main(argv=[], runner=tools, stream=io.StringIO())
    assert code == EXIT_TOOL_UNAVAILABLE
    assert "git" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Parsing units
# ---------------------------------------------------------------------------


def test_parse_worktrees_handles_branches_detached_and_backslashes() -> None:
    porcelain = (
        "worktree D:\\Github\\repo\nHEAD 1111111111111111111111111111111111111111\nbranch refs/heads/main\n"
        "\n"
        "worktree D:\\Github\\repo\\.claude\\worktrees\\a1\n"
        "HEAD 2222222222222222222222222222222222222222\ndetached\n"
    )
    parsed = parse_worktrees(porcelain)
    assert [w.path for w in parsed] == ["D:/Github/repo", "D:/Github/repo/.claude/worktrees/a1"]
    assert parsed[0].branch == "main"
    assert parsed[1].branch == ""
    assert parsed[1].head.startswith("2222")


def test_report_counts_every_alert_class_once() -> None:
    """A worktree can be broken in several ways at once; each one counts."""
    snapshot = Snapshot(
        worktrees=[Worktree(path="/tmp/wt", branch="feat/x", head="abc1234", dirty="1 modified")],
        pull_requests=[],
        open_issues=[],
        remote_branches=set(),
    )
    stream = io.StringIO()
    assert report(snapshot, stream=stream) == 3
    out = stream.getvalue()
    assert "uncommitted changes" in out
    assert "branch not pushed" in out
    assert "no open PR" in out


def test_unparseable_gh_json_raises(worktree_dir: Path) -> None:
    """Garbage on stdout is an error, not an empty list."""
    tools = _healthy(worktree_dir)
    original = tools.__call__

    def broken(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        if tuple(cmd)[: len(GH_PR_LIST)] == GH_PR_LIST:
            tools.calls.append(list(cmd))
            return SimpleNamespace(returncode=0, stdout="not json at all", stderr="")
        return original(cmd, **kwargs)

    with pytest.raises(AuditToolError, match="unparseable JSON"):
        audit(runner=broken, stream=io.StringIO())


# ---------------------------------------------------------------------------
# The ledger is gone and stays gone (#306 acceptance)
# ---------------------------------------------------------------------------


def test_task_ledger_file_is_deleted() -> None:
    """``docs/agent_tasks.json`` must not come back — GitHub is the state."""
    assert not (REPO_ROOT / "docs" / "agent_tasks.json").exists()


def test_audit_writes_no_state_file(worktree_dir: Path) -> None:
    """A full run leaves docs/ byte-identical: the audit is read-only."""
    docs = REPO_ROOT / "docs"
    before = sorted(p.name for p in docs.iterdir())
    _run_audit(_healthy(worktree_dir))
    assert sorted(p.name for p in docs.iterdir()) == before
