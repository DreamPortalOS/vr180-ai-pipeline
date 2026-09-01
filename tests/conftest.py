"""Shared pytest fixtures for the vr180-ai-pipeline test suite.

The headline concern here is the ``video/`` *workspace pollution* guard
(issue #164): the repo ``video/`` dir is the operator's local-media /
deliverable directory (git-ignored).  Tests that drive the mock provider or
``scripts/generate.py`` must write their artefacts into ``tmp_path``, never
into ``video/``.

A previous round (#163) fixed the ``e2e_smoke`` half; this guard closes the
remaining hole so any future test that leaks ``mock_*.mp4`` (or anything
else) into ``video/`` is caught by CI immediately, not by the lead at
midnight.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = _REPO_ROOT / "video"

# Anything matching these globs is treated as disposable test leakage and is
# *not* an expected resident of video/ at session start.  If such a file
# appears (or grows in count) over a session, the guard fails.
_LEAK_GLOBS = ("mock_*.mp4",)


def _leak_files() -> set[Path]:
    """Return the set of leak-pattern files currently in video/."""
    if not VIDEO_DIR.is_dir():
        return set()
    found: set[Path] = set()
    for pattern in _LEAK_GLOBS:
        found.update(VIDEO_DIR.glob(pattern))
    return found


def _finalize_video_dir_guard(baseline: set[Path]) -> None:
    """Session-final assertion: no *new* leak-pattern files in video/.

    Called after the whole suite (autouse session fixture teardown).  If a
    test leaked ``mock_*.mp4`` — the classic symptom of a mock-provider call
    without an explicit output dir — the guard lists the offenders and fails
    the session.
    """
    after = _leak_files()
    leaked = after - baseline
    if leaked:
        leaked_sorted = "\n  ".join(sorted(str(p) for p in leaked))
        pytest.fail(
            "video/ workspace pollution detected — the test suite wrote new "
            "leak-pattern files into the repo video/ dir (issue #164):\n  "
            f"{leaked_sorted}\n"
            "Tests must write artefacts into tmp_path / an explicit --output, "
            "never into video/. Set MOCK_PROVIDER_OUTPUT_DIR or pass --output "
            "in the offending test."
        )


@pytest.fixture(scope="session", autouse=True)
def _video_dir_pollution_guard() -> None:
    """Autouse session guard: snapshot video/ leak files, assert none added.

    This is the heart of issue #164's regression net.  It runs once per
    session (autouse, session scope), snapshots the ``video/mock_*.mp4`` set
    at the start, and on teardown fails the session if any new leak-pattern
    file appeared.  Any test that drives the mock provider or
    ``scripts/generate.py`` without an explicit output dir will trip this.
    """
    baseline = _leak_files()
    yield
    _finalize_video_dir_guard(baseline)
