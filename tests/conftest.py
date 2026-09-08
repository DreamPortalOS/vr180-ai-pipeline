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

Issue #227 (K-22): the depth (``models/.cache/depth``) and stereo
(``models/.cache/stereo``) product caches have their default root *in-repo*
on purpose (production use).  If any test instantiates the estimators /
renderers without an explicit ``cache_dir``, they write into the repo and
make the suite non-idempotent (first run green, second run green-on-hit-but-
wrong-path).  The function-scoped ``_cache_dir_redirect`` fixture below
redirects both module-level default-root constants to ``tmp_path`` for
every test function, so no test ever touches the repo cache.  The session
fixture ``_repo_cache_pollution_guard`` is the regression net: it snapshots
``models/.cache`` at session start and fails the session if any new entry
appears — catching the exact class of leakage that trips only on a dirty
local machine, not on a clean CI runner.

Issue #315 (W-5) generalises both of the above: ``_repo_root_pollution_guard``
snapshots the *repo root itself* and fails the session if the suite created
anything there.  ``video/`` and ``models/.cache`` were the two leaks somebody
happened to notice; a full run was in fact also leaving ``MagicMock/``,
``x_vr180_temp/``, ``o/``, ``.tmp_concat/`` and a real ``third_party/`` clone
behind.  ``git status`` never complained (empty directories are invisible to
git and ``*.mp4`` is ignored), which is exactly why the guard compares
directory listings instead of shelling out to git.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = _REPO_ROOT / "video"
_REPO_CACHE_DIR = _REPO_ROOT / "models" / ".cache"

# Anything matching these globs is treated as disposable test leakage and is
# *not* an expected resident of video/ at session start.  If such a file
# appears (or grows in count) over a session, the guard fails.
_LEAK_GLOBS = ("mock_*.mp4",)


# ---------------------------------------------------------------------------
# video/ pollution guard (issue #164)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# models/.cache pollution guard (issue #227, K-22)
# ---------------------------------------------------------------------------


def _repo_cache_entries() -> set[Path]:
    """Return the set of sub-entries currently under models/.cache.

    A "sub-entry" is any file or directory directly under the cache root
    (e.g. ``models/.cache/depth/<key>/`` or ``models/.cache/stereo/<key>/``).
    Intermediate cache-subdirs (``depth/``, ``stereo/``) count as entries too.
    """
    if not _REPO_CACHE_DIR.is_dir():
        return set()
    found: set[Path] = set()
    for entry in _REPO_CACHE_DIR.iterdir():
        found.add(entry)
        if entry.is_dir():
            for sub in entry.rglob("*"):
                found.add(sub)
    return found


def _finalize_repo_cache_guard(baseline: set[Path]) -> None:
    """Session-final assertion: no *new* entries in models/.cache.

    Any new file / directory under ``models/.cache`` that was not present at
    session start is treated as test pollution (issue #227): the suite is
    writing into the repo's production cache root instead of into tmp_path.
    """
    after = _repo_cache_entries()
    leaked = after - baseline
    if leaked:
        leaked_sorted = "\n  ".join(sorted(str(p) for p in leaked))
        pytest.fail(
            "models/.cache workspace pollution detected — the test suite "
            "wrote new entries into the repo's production cache root "
            "(issue #227, K-22):\n  "
            f"{leaked_sorted}\n"
            "Tests must instantiate StereoCrafterRenderer / "
            "DepthCrafterEstimator with cache_dir under tmp_path, or rely on "
            "the autouse _cache_dir_redirect fixture that overrides the "
            "module-level _DEFAULT_*_CACHE_DIR constants. Do NOT let tests "
            "hit the in-repo cache."
        )


@pytest.fixture(scope="session", autouse=True)
def _repo_cache_pollution_guard() -> None:
    """Autouse session guard: snapshot models/.cache, assert no new entries.

    Regression net for issue #227: a dirty local machine that runs the suite
    twice accumulates cache entries the first run around, so the second run
    hits the cache and returns a path under ``models/.cache/stereo`` instead
    of the expected tmp_path.  On a clean CI runner the first run never
    produces a hit, so this is invisible there.  This fixture catches any
    test that lets the repo cache be written into — locally or in CI.
    """
    baseline = _repo_cache_entries()
    yield
    _finalize_repo_cache_guard(baseline)


# ---------------------------------------------------------------------------
# repo-root pollution guard (issue #315, W-5)
# ---------------------------------------------------------------------------

#: Entries the session is *allowed* to create at the repo root.  These are
#: tool caches written by pytest / ruff / coverage themselves — never test
#: artefacts.  Everything else that appears during a session is leakage.
#:
#: Deliberately short.  Growing this set is how a guard turns into a rubber
#: stamp: the fix for a new name showing up is to stop the test writing it,
#: not to append it here.
_ROOT_GUARD_ALLOWED = frozenset(
    {
        "__pycache__",
        ".hypothesis",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "coverage.xml",
        "htmlcov",
    }
)


def _repo_root_entries(root: Path) -> set[str]:
    """Names of the entries directly under *root*, minus tool artefacts.

    Only the top level is inspected: that is where the leaks land (a relative
    path resolved against the repo-root cwd, or a mock repr used as a
    directory name), and it keeps the snapshot cheap enough to take twice per
    session even with a populated ``video/`` next door.
    """
    return {
        entry.name
        for entry in root.iterdir()
        if entry.name not in _ROOT_GUARD_ALLOWED and not entry.name.startswith(".coverage")
    }


def _finalize_repo_root_guard(root: Path, baseline: set[str]) -> None:
    """Session-final assertion: the suite created nothing at the repo root.

    Kept as a plain function taking *root* so it can be exercised directly
    against a throw-away directory (see ``tests/test_repo_root_guard.py``) —
    a session-teardown assertion that is never itself tested is a guard
    nobody can trust.
    """
    leaked = sorted(_repo_root_entries(root) - baseline)
    if leaked:
        leaked_list = "\n  ".join(leaked)
        pytest.fail(
            "repo-root pollution detected — the test suite created new entries "
            "in the repository root (issue #315, W-5):\n  "
            f"{leaked_list}\n"
            "Tests must never write inside the repo. The three usual causes:\n"
            "  * a MagicMock reached a path join (e.g. args.temp_dir) and its "
            "repr became a directory name — give the mock a real str(tmp_path);\n"
            "  * a subprocess inherited the repo-root cwd — pass cwd=tmp_path "
            "to subprocess.run, or monkeypatch.chdir(tmp_path);\n"
            "  * a relative --output / --outdir / --temp-dir was used — make it "
            "absolute under tmp_path.\n"
            "Do NOT .gitignore the offending name: that hides the leak instead "
            "of fixing it (and empty dirs are invisible to git anyway)."
        )


@pytest.fixture(scope="session", autouse=True)
def _repo_root_pollution_guard() -> None:
    """Autouse session guard: the repo root must look identical afterwards.

    Issue #315: a full run used to leave ``MagicMock/mock.temp_dir/<id>/concat``,
    ``x_vr180_temp/``, ``o/``, ``third_party/`` … behind.  ``git status`` stays
    clean for most of them (empty dirs are untracked-invisible, ``*.mp4`` is
    ignored), so nothing caught it — hence this snapshot-based guard rather
    than a git-based one.
    """
    baseline = _repo_root_entries(_REPO_ROOT)
    yield
    _finalize_repo_root_guard(_REPO_ROOT, baseline)


# ---------------------------------------------------------------------------
# Per-function cache-dir redirect (issue #227, K-22)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cache_dir_redirect(tmp_path, monkeypatch) -> None:
    """Redirect the depth and stereo *default* cache roots to tmp_path.

    Issue #227: both ``pipeline.depth_crafter`` and ``pipeline.stereo_crafter``
    expose a module-level constant (``_DEFAULT_DEPTH_CACHE_DIR`` and
    ``_DEFAULT_STEREO_CACHE_DIR``) that points at ``models/.cache/{depth,stereo}``
    — the production in-repo default.  Any test that builds a renderer /
    estimator without an explicit ``cache_dir`` would otherwise write there,
    breaking idempotency (hit on run 2) and polluting the repo.

    This autouse, function-scoped fixture monkeypatches both constants to a
    fresh tmp_path subdirectory for every test function.  Tests that *do*
    pass an explicit ``cache_dir`` (the cache-dedicated tests) are unaffected:
    their ``cache_dir`` arg wins over the module constant.

    The patch is imported fresh here so the fixture does not itself depend on
    the pipeline package being importable at collection time on a machine
    lacking the heavy GPU stack (CI is CPU-only).
    """
    from pipeline import depth_crafter, stereo_crafter

    # Use names that do not shadow a test's own tmp_path subpaths (e.g. a
    # test's ``cache_dir=tmp_path / "cache"`` must not be a parent of the
    # redirected default, otherwise the redirected subdirs appear as extra
    # children and trip "len(entries) == 1" assertions in the cache-dedicated
    # tests).  Sibling names under a distinct top-level are safe.
    depth_target = tmp_path / "_conftest_default_depth_cache"
    depth_target.mkdir(parents=True)
    stereo_target = tmp_path / "_conftest_default_stereo_cache"
    stereo_target.mkdir(parents=True)

    monkeypatch.setattr(depth_crafter, "_DEFAULT_DEPTH_CACHE_DIR", depth_target, raising=True)
    monkeypatch.setattr(stereo_crafter, "_DEFAULT_STEREO_CACHE_DIR", stereo_target, raising=True)
