"""W-5 (#315): self-tests for the repo-root pollution guard in conftest.py.

The guard is a *session teardown* assertion, so nothing in a normal run ever
proves it still works: if it silently stopped detecting leaks, every suite
would still be green and the repo would quietly fill up with
``MagicMock/mock.temp_dir/<id>/concat`` again.  A guard nobody tests is a
guard nobody can trust, so this file pins it from both ends:

* unit level — :func:`tests.conftest._finalize_repo_root_guard` is called
  directly against a throw-away directory, so the pass/fail decision and the
  wording of the failure are asserted without waiting for a whole session;
* session level — a real nested ``pytest`` run in a temp directory, wired to
  the very same guard functions, with one test that deliberately creates a
  directory at that fake "repo root".  That run must come back **red**, which
  is the acceptance evidence the card asks for.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import _finalize_repo_root_guard, _repo_root_entries

REPO_ROOT = Path(__file__).resolve().parent.parent

#: ``pytest.fail`` raises ``_pytest.outcomes.Failed``, reachable publicly here.
Failed = pytest.fail.Exception


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    """A throw-away stand-in for the repo root.

    A *sub*directory of ``tmp_path``, not ``tmp_path`` itself: the autouse
    ``_cache_dir_redirect`` fixture already puts two directories in there, and
    the entry-set assertions below are exact.
    """
    root = tmp_path / "fake_repo"
    root.mkdir()
    return root


# ---------------------------------------------------------------------------
# _repo_root_entries: what counts as an entry
# ---------------------------------------------------------------------------


class TestRepoRootEntries:
    def test_lists_files_and_directories(self, fake_root: Path):
        (fake_root / "a_dir").mkdir()
        (fake_root / "a_file.txt").write_text("x", encoding="utf-8")
        assert _repo_root_entries(fake_root) == {"a_dir", "a_file.txt"}

    def test_tool_caches_are_not_entries(self, fake_root: Path):
        """pytest/ruff/coverage write these by design — they are not leakage."""
        for name in (".pytest_cache", ".ruff_cache", "__pycache__", "htmlcov"):
            (fake_root / name).mkdir()
        assert _repo_root_entries(fake_root) == set()

    def test_coverage_data_files_are_not_entries(self, fake_root: Path):
        """``.coverage`` and its parallel-mode ``.coverage.<host>.<pid>`` siblings."""
        (fake_root / ".coverage").write_text("", encoding="utf-8")
        (fake_root / ".coverage.runner.4711").write_text("", encoding="utf-8")
        assert _repo_root_entries(fake_root) == set()

    def test_only_the_top_level_is_inspected(self, fake_root: Path):
        """Nested content does not multiply the entry set — one name per child."""
        (fake_root / "top" / "deep" / "deeper").mkdir(parents=True)
        assert _repo_root_entries(fake_root) == {"top"}


# ---------------------------------------------------------------------------
# _finalize_repo_root_guard: the pass/fail decision
# ---------------------------------------------------------------------------


class TestFinalizeRepoRootGuard:
    def test_unchanged_root_passes(self, fake_root: Path):
        (fake_root / "pipeline").mkdir()
        baseline = _repo_root_entries(fake_root)
        _finalize_repo_root_guard(fake_root, baseline)  # must not raise

    def test_new_directory_fails(self, fake_root: Path):
        baseline = _repo_root_entries(fake_root)
        (fake_root / "MagicMock").mkdir()
        with pytest.raises(Failed) as ei:
            _finalize_repo_root_guard(fake_root, baseline)
        assert "repo-root pollution detected" in str(ei.value)
        assert "MagicMock" in str(ei.value)

    def test_new_file_fails_too(self, fake_root: Path):
        """A stray file at the root is the same bug as a stray directory."""
        baseline = _repo_root_entries(fake_root)
        (fake_root / "out.mp4").write_text("", encoding="utf-8")
        with pytest.raises(Failed) as ei:
            _finalize_repo_root_guard(fake_root, baseline)
        assert "out.mp4" in str(ei.value)

    def test_every_offender_is_named(self, fake_root: Path):
        baseline = _repo_root_entries(fake_root)
        for name in ("MagicMock", "x_vr180_temp", "o"):
            (fake_root / name).mkdir()
        with pytest.raises(Failed) as ei:
            _finalize_repo_root_guard(fake_root, baseline)
        message = str(ei.value)
        for name in ("MagicMock", "x_vr180_temp", "o"):
            assert name in message

    def test_disappearing_entries_are_not_a_failure(self, fake_root: Path):
        """The guard is about *additions*; a removed entry is somebody else's
        problem and must not turn into a confusing pollution report."""
        (fake_root / "scratch").mkdir()
        baseline = _repo_root_entries(fake_root)
        (fake_root / "scratch").rmdir()
        _finalize_repo_root_guard(fake_root, baseline)  # must not raise

    def test_new_tool_cache_is_not_a_failure(self, fake_root: Path):
        baseline = _repo_root_entries(fake_root)
        (fake_root / ".pytest_cache").mkdir()
        _finalize_repo_root_guard(fake_root, baseline)  # must not raise

    def test_failure_message_forbids_the_gitignore_shortcut(self, fake_root: Path):
        """#315 is explicit that hiding the leak behind .gitignore is wrong —
        the message has to say so, because that is the tempting 'fix'."""
        baseline = _repo_root_entries(fake_root)
        (fake_root / "leaked").mkdir()
        with pytest.raises(Failed) as ei:
            _finalize_repo_root_guard(fake_root, baseline)
        assert ".gitignore" in str(ei.value)


# ---------------------------------------------------------------------------
# End-to-end: a whole nested session really does go red
# ---------------------------------------------------------------------------

_NESTED_CONFTEST = """\
from pathlib import Path

import pytest

from tests.conftest import _finalize_repo_root_guard, _repo_root_entries

FAKE_ROOT = Path(__file__).resolve().parent


@pytest.fixture(scope="session", autouse=True)
def _repo_root_pollution_guard():
    baseline = _repo_root_entries(FAKE_ROOT)
    yield
    _finalize_repo_root_guard(FAKE_ROOT, baseline)
"""

_NESTED_TEST = """\
from pathlib import Path

FAKE_ROOT = Path(__file__).resolve().parent


def test_that_leaks_a_directory():
    (FAKE_ROOT / "MagicMock").mkdir(exist_ok=True)
    assert True
"""


@pytest.fixture(scope="module")
def nested_run(tmp_path_factory) -> subprocess.CompletedProcess[str]:
    """Run a real pytest session, guarded exactly like this suite is, whose
    single test creates a directory at its (fake) repo root.

    The nested session's own root is the temp directory — the guard is given
    that path explicitly — so this proves the wiring without touching the real
    repo root, and without depending on the outer session's own guard.
    """
    sandbox = tmp_path_factory.mktemp("nested_guard")
    (sandbox / "conftest.py").write_text(_NESTED_CONFTEST, encoding="utf-8")
    (sandbox / "test_leaky.py").write_text(_NESTED_TEST, encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", str(sandbox)],
        capture_output=True,
        text=True,
        cwd=str(sandbox),
        env=env,
        timeout=300,
    )


class TestGuardFailsARealSession:
    def test_session_exits_non_zero(self, nested_run):
        assert nested_run.returncode != 0, (
            f"the repo-root guard did NOT fail a session that leaked a directory:\n{nested_run.stdout[-2000:]}"
        )

    def test_report_names_the_leak_and_the_issue(self, nested_run):
        combined = nested_run.stdout + nested_run.stderr
        assert "repo-root pollution detected" in combined
        assert "MagicMock" in combined
        assert "#315" in combined

    def test_the_leaky_test_itself_still_passed(self, nested_run):
        """The guard fails at *teardown*, not by breaking the test — so the
        report stays readable: 1 passed, then an errored session."""
        assert "1 passed" in nested_run.stdout

    def test_proof_uses_the_real_guard_not_a_copy(self):
        """If the nested conftest ever stopped importing the real functions the
        end-to-end proof would degrade into testing a copy of the code."""
        assert "from tests.conftest import _finalize_repo_root_guard, _repo_root_entries" in _NESTED_CONFTEST
